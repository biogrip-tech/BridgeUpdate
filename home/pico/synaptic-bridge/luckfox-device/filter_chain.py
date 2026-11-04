"""
filter_chain.py — Motor de filtros IIR para streaming en tiempo real.

Diseñado para correr DENTRO del puente UDP->LSL en el Luckfox: recibe una
ventana (n_samples, n_channels) por llamada y la filtra en el sitio, manteniendo
el estado interno de cada filtro (zi) entre llamadas para que no haya
discontinuidades entre ventanas consecutivas.

IMPORTANTE — por qué no usar scipy.signal.filtfilt:
    filtfilt es de fase cero pero necesita ver la señal COMPLETA (futuro y
    pasado) para cada muestra: es un filtro "offline". En streaming solo
    tenemos el pasado, así que hay que usar un filtro CAUSAL con memoria
    (sosfilt + estado zi) que se actualiza ventana a ventana. La alternativa
    ingenua de "filtrar cada ventana de forma independiente" introduce un
    transitorio (una discontinuidad) al principio de cada ventana: eso es
    exactamente el estado (zi) que este módulo carga y guarda para evitarlo.

Familias soportadas: butter, cheby1, cheby2, ellip
Tipos: highpass, lowpass, notch (banda eliminada estrecha vía iirnotch)

Uso:
    cfg = {
        "highpass": {"enabled": True, "family": "butter", "order": 4, "cutoff": 20},
        "lowpass":  {"enabled": True, "family": "butter", "order": 4, "cutoff": 450},
        "notch":    {"enabled": True, "freq": 60, "quality": 30},
    }
    chain = FilterChain(fs=1000, n_channels=8, config=cfg)
    filtered_window = chain.process(raw_window)   # (n_samples, n_channels)
"""

import numpy as np
from scipy import signal

VALID_FAMILIES = {"butter", "cheby1", "cheby2", "ellip"}


def _design_sos(family: str, order: int, cutoff, fs: float, btype: str,
                 ripple: float = 1.0, atten: float = 40.0):
    """Diseña un filtro IIR y devuelve secciones de segundo orden (SOS).

    SOS (en vez de coeficientes b/a directos) porque es numéricamente mucho
    más estable para órdenes altos: b/a con orden >4-6 puede volverse
    inestable por errores de redondeo en punto flotante.
    """
    nyq = fs / 2.0
    wn = np.asarray(cutoff, dtype=float) / nyq
    # Cordura: la frecuencia normalizada debe quedar estrictamente en (0, 1)
    wn = np.clip(wn, 1e-6, 0.999999)

    if family == "butter":
        sos = signal.butter(order, wn, btype=btype, output="sos")
    elif family == "cheby1":
        sos = signal.cheby1(order, ripple, wn, btype=btype, output="sos")
    elif family == "cheby2":
        sos = signal.cheby2(order, atten, wn, btype=btype, output="sos")
    elif family == "ellip":
        sos = signal.ellip(order, ripple, atten, wn, btype=btype, output="sos")
    else:
        raise ValueError(f"Familia de filtro desconocida: {family}")
    return sos


def _design_notch_sos(freq: float, quality: float, fs: float):
    """Notch (banda eliminada estrecha) vía iirnotch, convertido a SOS."""
    nyq = fs / 2.0
    w0 = np.clip(freq / nyq, 1e-6, 0.999999)
    b, a = signal.iirnotch(w0, quality)
    sos = signal.tf2sos(b, a)
    return sos


class _Stage:
    """Una etapa de filtro (p.ej. highpass) con estado zi por canal."""

    def __init__(self, sos: np.ndarray, n_channels: int):
        self.sos = sos
        # zi por canal: shape (n_secciones, 2, n_channels)
        zi_single = signal.sosfilt_zi(sos)                       # (n_secciones, 2)
        self.zi = np.repeat(zi_single[:, :, None], n_channels, axis=2).astype(np.float64)

    def apply(self, window: np.ndarray) -> np.ndarray:
        """window: (n_samples, n_channels) -> filtra por columna, actualiza self.zi."""
        out, self.zi = signal.sosfilt(self.sos, window, axis=0, zi=self.zi)
        return out


class FilterChain:
    """Cadena de filtros: highpass -> lowpass -> notch, cada uno opcional.

    Si ninguno está habilitado, `process()` devuelve la señal cruda sin tocar
    (copia), tal como pediste: sin filtro seleccionado, se manda cruda.
    """

    def __init__(self, fs: float, n_channels: int, config: dict):
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.stages = []          # lista de (nombre, _Stage)
        self.errors = []          # avisos de diseño (p.ej. cutoff inválido)
        self._build(config or {})

    def _build(self, config: dict):
        hp = config.get("highpass", {})
        lp = config.get("lowpass", {})
        nt = config.get("notch", {})

        if hp.get("enabled"):
            try:
                sos = _design_sos(
                    family=hp.get("family", "butter"),
                    order=int(hp.get("order", 4)),
                    cutoff=float(hp["cutoff"]),
                    fs=self.fs, btype="highpass",
                    ripple=float(hp.get("ripple", 1.0)),
                    atten=float(hp.get("attenuation", 40.0)),
                )
                self.stages.append(("highpass", _Stage(sos, self.n_channels)))
            except Exception as e:
                self.errors.append(f"highpass: {e}")

        if lp.get("enabled"):
            try:
                sos = _design_sos(
                    family=lp.get("family", "butter"),
                    order=int(lp.get("order", 4)),
                    cutoff=float(lp["cutoff"]),
                    fs=self.fs, btype="lowpass",
                    ripple=float(lp.get("ripple", 1.0)),
                    atten=float(lp.get("attenuation", 40.0)),
                )
                self.stages.append(("lowpass", _Stage(sos, self.n_channels)))
            except Exception as e:
                self.errors.append(f"lowpass: {e}")

        if nt.get("enabled"):
            try:
                sos = _design_notch_sos(
                    freq=float(nt.get("freq", 60.0)),
                    quality=float(nt.get("quality", 30.0)),
                    fs=self.fs,
                )
                self.stages.append(("notch", _Stage(sos, self.n_channels)))
            except Exception as e:
                self.errors.append(f"notch: {e}")

    @property
    def active(self) -> bool:
        return len(self.stages) > 0

    def process(self, window: np.ndarray) -> np.ndarray:
        """window: (n_samples, n_channels) float. Devuelve la señal filtrada
        (o una copia cruda si no hay etapas activas)."""
        if not self.stages:
            return window.copy()
        out = window.astype(np.float64, copy=True)
        for _name, stage in self.stages:
            out = stage.apply(out)
        return out

    def describe(self) -> list:
        """Resumen legible de la cadena activa, para logs/estado."""
        return [name for name, _ in self.stages]
