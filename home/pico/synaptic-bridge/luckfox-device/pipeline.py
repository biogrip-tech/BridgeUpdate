"""
pipeline.py — Pipeline de procesamiento por etapas para el puente UDP->LSL (v2).

Un pipeline es una LISTA ORDENADA de etapas. Cada etapa tiene un tipo, sus
parámetros y, opcionalmente, los canales a los que se aplica:

    "pipeline": [
        {"type": "highpass", "family": "butter", "order": 4, "cutoff": 20},
        {"type": "notch", "freq": 60, "quality": 30, "harmonics": 2, "channels": [1, 2, 3]},
        {"type": "car", "channels": "all"},
        {"type": "whitening", "time_constant": 2.0, "reference": "subj01_ref.npz"},
        {"type": "custom", "file": "envolvente_rms.py", "params": {"window_ms": 100}}
    ]

Canales: "all" (por defecto) o una lista de números EMPEZANDO EN 1, igual que
las etiquetas EMG_1..EMG_N del stream. Los canales no seleccionados pasan sin
cambios por esa etapa.

Tipos disponibles:
    highpass, lowpass     -> cutoff (Hz), family, order [, ripple, attenuation]
    bandpass, bandstop    -> low, high (Hz), family, order [, ripple, attenuation]
    notch                 -> freq (Hz), quality [, harmonics]
    car                   -> referencia promedio común sobre los canales elegidos
    whitening             -> blanqueo espacial adaptativo (StreamingWhitener de
                             euclidean_align.py), opcionalmente arrancando desde
                             una referencia .npz guardada
    custom (o plugin)     -> función de Python propia (ver plugins/)

Cualquier etapa admite "enabled": false para desactivarla sin borrarla, y
"label" para darle un nombre legible en logs y metadatos.

Robustez: si una etapa lanza una excepción, devuelve una forma incorrecta o
produce NaN/Inf, se DESACTIVA (pasa la señal sin cambios), se registra el
error y el resto del pipeline sigue funcionando. Los datos nunca se detienen
por culpa de una etapa.

Compatible con numpy 1.21 y scipy 1.8 (las versiones de Ubuntu 22.04 en el
Luckfox).
"""

import importlib.util
import logging
import os
import time

import numpy as np
from scipy import signal

from filter_chain import VALID_FAMILIES, _design_notch_sos, _design_sos

log = logging.getLogger("pipeline")


class StageError(Exception):
    """Error de configuración o de ejecución de una etapa."""


def resolve_channels(spec, n_channels):
    """Convierte "all" / int / lista (base 1) en un array de índices base 0."""
    if spec is None or spec == "all":
        return np.arange(n_channels)
    if isinstance(spec, (int, float)):
        spec = [spec]
    if not isinstance(spec, (list, tuple)):
        raise StageError(f"'channels' debe ser \"all\" o una lista de números, no {spec!r}")
    idx = []
    for c in spec:
        c = int(c)
        if not 1 <= c <= n_channels:
            raise StageError(f"canal {c} fuera de rango (1..{n_channels})")
        idx.append(c - 1)
    if not idx:
        raise StageError("la lista de canales está vacía")
    return np.array(sorted(set(idx)), dtype=int)


def _resolve_path(path, base_dir):
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Etapas
# ═══════════════════════════════════════════════════════════════════════════
class Stage:
    """Etapa base: gestiona la selección de canales y las estadísticas."""

    type_name = "base"

    def __init__(self, fs, n_channels, params, ctx):
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.params = params
        self.ctx = ctx
        self.channels = resolve_channels(params.get("channels", "all"), n_channels)
        self.all_channels = len(self.channels) == self.n_channels
        self.label = str(params.get("label") or params.get("type") or self.type_name)
        # Estadísticas y estado de error
        self.calls = 0
        self.total_s = 0.0
        self.max_s = 0.0
        self.disabled = False
        self.error = None

    @property
    def n_selected(self):
        return len(self.channels)

    def process(self, x):
        """x: (n_muestras, n_canales) float64. Devuelve el array procesado."""
        if self.all_channels:
            return self._process(x)
        sub = np.ascontiguousarray(x[:, self.channels])
        out = self._process(sub)
        x[:, self.channels] = out
        return x

    def _process(self, x):
        raise NotImplementedError

    def describe(self):
        if self.all_channels:
            ch = "all"
        else:
            ch = ",".join(str(c + 1) for c in self.channels)
        return f"{self.label}[{ch}]"

    def watched_files(self):
        """Archivos cuyo cambio debe recargar esta etapa (plugins, referencias)."""
        return []

    def stats(self):
        avg_ms = 1000.0 * self.total_s / self.calls if self.calls else 0.0
        return {
            "stage": self.describe(),
            "avg_ms": round(avg_ms, 3),
            "max_ms": round(1000.0 * self.max_s, 3),
            "calls": self.calls,
            "disabled": self.disabled,
            "error": self.error,
        }


class _IIRBase(Stage):
    """Filtro IIR causal en SOS con estado (zi) persistente por canal."""

    def _init_sos(self, sos):
        self.sos = sos
        self.zi = None   # se inicializa con la primera ventana

    def _process(self, x):
        if self.zi is None:
            # Estado inicial en régimen permanente para el primer valor de cada
            # canal: evita el transitorio grande del arranque (sobre todo con
            # offset de DC antes del pasa-altas).
            zi_unit = signal.sosfilt_zi(self.sos)               # (secciones, 2)
            self.zi = zi_unit[:, :, None] * x[0][None, None, :]
        out, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return out


class IIRStage(_IIRBase):
    """highpass / lowpass / bandpass / bandstop."""

    def __init__(self, fs, n_channels, params, ctx):
        super().__init__(fs, n_channels, params, ctx)
        btype = params["type"]
        family = params.get("family", "butter")
        if family not in VALID_FAMILIES:
            raise StageError(f"familia '{family}' no válida; usa {sorted(VALID_FAMILIES)}")
        order = int(params.get("order", 4))
        if not 1 <= order <= 12:
            raise StageError(f"orden {order} fuera de rango (1..12)")
        nyq = self.fs / 2.0

        if btype in ("highpass", "lowpass"):
            if "cutoff" not in params:
                raise StageError("falta 'cutoff' (Hz)")
            cutoff = float(params["cutoff"])
            if not 0.0 < cutoff < nyq:
                raise StageError(f"cutoff={cutoff} Hz debe estar entre 0 y fs/2={nyq:g} Hz")
            desc = f"{cutoff:g}Hz"
        else:
            if "low" not in params or "high" not in params:
                raise StageError("faltan 'low' y/o 'high' (Hz)")
            low, high = float(params["low"]), float(params["high"])
            if not 0.0 < low < high < nyq:
                raise StageError(f"se necesita 0 < low < high < fs/2={nyq:g} Hz (low={low}, high={high})")
            cutoff = [low, high]
            desc = f"{low:g}-{high:g}Hz"

        sos = _design_sos(family=family, order=order, cutoff=cutoff, fs=self.fs, btype=btype,
                          ripple=float(params.get("ripple", 1.0)),
                          atten=float(params.get("attenuation", 40.0)))
        self._init_sos(sos)
        if not params.get("label"):
            self.label = f"{btype}({family},{order},{desc})"


class NotchStage(_IIRBase):
    """Notch en `freq` y, opcionalmente, en sus armónicos (2·freq, 3·freq...)."""

    def __init__(self, fs, n_channels, params, ctx):
        super().__init__(fs, n_channels, params, ctx)
        freq = float(params.get("freq", 60.0))
        quality = float(params.get("quality", 30.0))
        harmonics = int(params.get("harmonics", 1))
        nyq = self.fs / 2.0
        if not 0.0 < freq < nyq:
            raise StageError(f"freq={freq} Hz debe estar entre 0 y fs/2={nyq:g} Hz")
        if quality <= 0:
            raise StageError("quality debe ser > 0")
        sections = []
        for k in range(1, max(1, harmonics) + 1):
            f = freq * k
            if f >= nyq:
                break
            sections.append(_design_notch_sos(f, quality, self.fs))
        self._init_sos(np.vstack(sections))
        if not params.get("label"):
            h = f"x{len(sections)}" if len(sections) > 1 else ""
            self.label = f"notch({freq:g}Hz{h},Q{quality:g})"


class CARStage(Stage):
    """Referencia promedio común: resta a cada canal la media de los elegidos.

    Conviene ponerla DESPUÉS del pasa-altas: si un electrodo tiene offset de
    DC, la media lo arrastraría a todos los demás canales.
    """

    def __init__(self, fs, n_channels, params, ctx):
        super().__init__(fs, n_channels, params, ctx)
        if self.n_selected < 2:
            raise StageError("CAR necesita al menos 2 canales")
        if not params.get("label"):
            self.label = "car"

    def _process(self, x):
        return x - x.mean(axis=1, keepdims=True)


class WhiteningStage(Stage):
    """Blanqueo espacial adaptativo (StreamingWhitener de euclidean_align.py).

    Parámetros:
        time_constant  memoria en segundos (por defecto 2.0)
        refresh_every  muestras entre recálculos de la matriz (por defecto 25)
        shrinkage      regularización hacia la identidad (por defecto 1e-3)
        subtract_mean  quitar la media móvil (por defecto false)
        reference      .npz guardado con EuclideanAlignment.save() para arrancar
                       ya alineado (ruta relativa a la carpeta de la config)
        adapt          true (por defecto) = sigue adaptándose; false = usa la
                       referencia fija (requiere 'reference')

    Ojo: mezcla canales. Tras esta etapa, el "canal 3" ya no es el electrodo 3
    sino una combinación de los canales seleccionados.
    """

    def __init__(self, fs, n_channels, params, ctx):
        super().__init__(fs, n_channels, params, ctx)
        try:
            from euclidean_align import StreamingWhitener
        except ImportError as e:
            raise StageError("no se encuentra euclidean_align.py junto al puente") from e
        if self.n_selected < 2:
            raise StageError("whitening necesita al menos 2 canales")
        self.adapt = bool(params.get("adapt", True))
        self.whitener = StreamingWhitener(
            fs=self.fs,
            time_constant=float(params.get("time_constant", 2.0)),
            refresh_every=int(params.get("refresh_every", 25)),
            shrinkage=float(params.get("shrinkage", 1e-3)),
            subtract_mean=bool(params.get("subtract_mean", False)),
        )
        self.reference_path = None
        ref = params.get("reference")
        if ref:
            self.reference_path = _resolve_path(ref, ctx.get("data_dir", "."))
            if not os.path.exists(self.reference_path):
                raise StageError(f"no existe la referencia {self.reference_path}")
            d = np.load(self.reference_path)
            R = np.asarray(d["R_bar"], dtype=float)
            if R.shape != (self.n_selected, self.n_selected):
                raise StageError(f"la referencia es de {R.shape[0]} canales y la etapa usa {self.n_selected}")
            self.whitener.seed(R)
        elif not self.adapt:
            raise StageError("adapt=false necesita una 'reference'")
        if not params.get("label"):
            self.label = "whitening" + ("" if self.adapt else "(fijo)")

    def _process(self, x):
        if self.adapt:
            return self.whitener.partial_fit_transform(x)
        return self.whitener.transform(x)

    def watched_files(self):
        return [self.reference_path] if self.reference_path else []


class PluginStage(Stage):
    """Función de Python propia, cargada desde un archivo .py.

    Contrato del plugin:

        def init(fs, n_channels, params):      # opcional
            return estado                       # cualquier objeto (dict, etc.)

        def process(window, state):            # obligatorio
            return nueva_ventana                # misma forma que window

    `window` es (n_muestras, n_canales_seleccionados) en float64 y es una
    copia: el plugin puede modificarla sin efectos secundarios. `params` es el
    diccionario "params" de la etapa en la config.
    """

    def __init__(self, fs, n_channels, params, ctx):
        super().__init__(fs, n_channels, params, ctx)
        if not params.get("file"):
            raise StageError("falta 'file' con el nombre del plugin (.py)")
        self.path = _resolve_path(params["file"], ctx.get("plugins_dir", "."))
        if not os.path.exists(self.path):
            raise StageError(f"no existe el plugin {self.path}")
        mod_name = "sb_plugin_" + os.path.splitext(os.path.basename(self.path))[0] + f"_{id(self)}"
        spec = importlib.util.spec_from_file_location(mod_name, self.path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise StageError(f"error al importar {os.path.basename(self.path)}: {e}") from e
        if not callable(getattr(module, "process", None)):
            raise StageError(f"{os.path.basename(self.path)} no define process(window, state)")
        self.module = module
        user_params = params.get("params") or {}
        init_fn = getattr(module, "init", None)
        try:
            self.state = init_fn(self.fs, self.n_selected, user_params) if callable(init_fn) else None
        except Exception as e:
            raise StageError(f"init() falló: {e}") from e
        if not params.get("label"):
            self.label = "custom:" + os.path.basename(self.path)

    def _process(self, x):
        out = self.module.process(x.copy(), self.state)
        out = np.asarray(out, dtype=np.float64)
        if out.shape != x.shape:
            raise StageError(f"process() devolvió forma {out.shape}; se esperaba {x.shape}")
        return out

    def watched_files(self):
        return [self.path]


STAGE_TYPES = {
    "highpass": IIRStage,
    "lowpass": IIRStage,
    "bandpass": IIRStage,
    "bandstop": IIRStage,
    "notch": NotchStage,
    "car": CARStage,
    "whitening": WhiteningStage,
    "custom": PluginStage,
    "plugin": PluginStage,
}


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════
class Pipeline:
    def __init__(self, fs, n_channels, specs, ctx=None):
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.ctx = ctx or {}
        self.stages = []
        self.errors = []       # errores al construir (etapas que no se crearon)
        for i, spec in enumerate(specs or []):
            if not isinstance(spec, dict):
                self.errors.append(f"etapa {i + 1}: debe ser un objeto JSON")
                continue
            if not spec.get("enabled", True):
                continue
            stype = spec.get("type")
            cls = STAGE_TYPES.get(stype)
            if cls is None:
                self.errors.append(f"etapa {i + 1}: tipo '{stype}' desconocido; usa {sorted(STAGE_TYPES)}")
                continue
            try:
                self.stages.append(cls(self.fs, self.n_channels, spec, self.ctx))
            except Exception as e:
                self.errors.append(f"etapa {i + 1} ({stype}): {e}")

    @property
    def active(self):
        return any(not s.disabled for s in self.stages)

    def process(self, window):
        """window: (n_muestras, n_canales). Devuelve float64 procesado."""
        out = np.array(window, dtype=np.float64, copy=True, order="C")
        for st in self.stages:
            if st.disabled:
                continue
            t0 = time.perf_counter()
            try:
                res = st.process(out)
                if not np.all(np.isfinite(res)):
                    raise StageError("produjo NaN o Inf")
            except Exception as e:
                # Se desactiva SOLO esta etapa; la señal sigue sin ese paso.
                st.disabled = True
                st.error = str(e)
                log.error("Etapa '%s' desactivada por error: %s", st.describe(), e)
                continue
            dt = time.perf_counter() - t0
            st.calls += 1
            st.total_s += dt
            if dt > st.max_s:
                st.max_s = dt
            out = res
        return out

    def describe(self):
        return [s.describe() for s in self.stages if not s.disabled]

    def stats(self):
        return [s.stats() for s in self.stages]

    def watched_files(self):
        files = {}
        for s in self.stages:
            for p in s.watched_files():
                try:
                    files[p] = os.path.getmtime(p)
                except OSError:
                    files[p] = None
        return files


def legacy_filters_to_pipeline(filters):
    """Convierte el formato antiguo {"highpass":{...}, "lowpass":{...},
    "notch":{...}} (el que escribe la GUI actual) en una lista de etapas."""
    specs = []
    for name in ("highpass", "lowpass"):
        f = filters.get(name) or {}
        if f.get("enabled"):
            spec = {"type": name}
            spec.update({k: v for k, v in f.items() if k != "enabled"})
            specs.append(spec)
    nt = filters.get("notch") or {}
    if nt.get("enabled"):
        specs.append({"type": "notch", "freq": nt.get("freq", 60), "quality": nt.get("quality", 30)})
    return specs
