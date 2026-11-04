"""
lsl_stream.py — Acceso headless (sin GUI) a un stream LSL de EMG.

Pensado para usarse desde cualquier otro código: conecta a un stream LSL
(por nombre, tipo o source_id), lee las muestras en un hilo de fondo y las
deja disponibles de tres formas, según lo que necesites:

  1) Buffer circular (polling)  -> get_channel(), snapshot(), get_latest()
     Ideal para "dame los últimos N segundos" cada cierto tiempo.

  2) Cola de no-leídos (pull)    -> pull()
     Devuelve solo lo nuevo desde la última llamada. Útil para procesar por
     bloques sin repetir muestras.

  3) Callback por chunk          -> on_data=fn
     Se llama con cada bloque en cuanto llega. La opción sin pérdidas para
     grabar o extraer características en tiempo real.

No dibuja nada. Solo datos.

Dependencias:
    pip install pylsl          (obligatorio)
    pip install numpy          (opcional; habilita salida como np.ndarray)

Ejemplo mínimo:
    from lsl_stream import LSLStream

    with LSLStream(name="BioMyo_EMG") as emg:
        emg.wait_until_ready(timeout=5)
        times, data = emg.get_latest(500)      # data: (muestras, canales)
        print(emg.info.n_channels, "canales @", emg.measured_rate(), "Hz")
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from pylsl import StreamInlet, resolve_streams, proc_clocksync, proc_dejitter
import socket

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:  # numpy es opcional
    np = None
    HAS_NUMPY = False

log = logging.getLogger("lsl_stream")

DEFAULT_FS = 500.0
DEFAULT_BUFFER_SECS = 10.0


# ═══════════════════════════════════════════════════════════════════════════
# Descubrimiento de streams
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class StreamDescription:
    """Metadatos de un stream visible en la red (antes de conectarse)."""
    name: str
    type: str
    n_channels: int
    nominal_srate: float
    source_id: str
    uid: str
    _info: object = field(default=None, repr=False)  # StreamInfo de pylsl


@dataclass
class StreamInfoData:
    """Info del stream ya conectado."""
    name: str
    type: str
    source_id: str
    n_channels: int
    nominal_srate: float
    channel_names: List[str]


def list_streams(timeout: float = 2.0) -> List[StreamDescription]:
    """Devuelve todos los streams LSL visibles en la red."""
    out: List[StreamDescription] = []
    try:
        for info in resolve_streams(wait_time=timeout):
            out.append(StreamDescription(
                name=info.name(),
                type=info.type(),
                n_channels=info.channel_count(),
                nominal_srate=info.nominal_srate(),
                source_id=info.source_id(),
                uid=info.uid(),
                _info=info,
            ))
    except Exception as e:
        log.error("Error resolviendo streams: %s", e)
    return out


def find_stream(name: Optional[str] = None, stype: Optional[str] = None,
                source_id: Optional[str] = None,
                timeout: float = 2.0) -> Optional[StreamDescription]:
    """Primer stream que cumpla los filtros dados (None = no filtra por ese campo)."""
    for d in list_streams(timeout):
        if name and d.name != name:
            continue
        if stype and d.type != stype:
            continue
        if source_id and d.source_id != source_id:
            continue
        return d
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Cliente del stream
# ═══════════════════════════════════════════════════════════════════════════
class LSLStream:
    """Conecta y lee un único stream LSL en segundo plano.

    Parámetros de selección (para connect()):
        name / stype / source_id : filtros; el primero que coincida se usa.
    Otros:
        buffer_secs   : segundos de historia en el buffer circular.
        max_channels  : recorta a N canales (None = todos los del stream).
        on_data       : callback(samples, timestamps) por cada chunk (opcional).
        recover       : reintentar automáticamente si el stream se cae.
        keep_unpulled : mantener la cola para pull() (desactívala si solo usas
                        callback o polling, para ahorrar algo de memoria).
    """

    def __init__(self, name: Optional[str] = None, stype: Optional[str] = None,
                 source_id: Optional[str] = None,
                 buffer_secs: float = DEFAULT_BUFFER_SECS,
                 max_channels: Optional[int] = None,
                 on_data: Optional[Callable[[list, list], None]] = None,
                 recover: bool = True, keep_unpulled: bool = True):
        self._name = name
        self._stype = stype
        self._source_id = source_id
        self.buffer_secs = float(buffer_secs)
        self._max_channels = max_channels
        self._on_data = on_data
        self._recover = recover
        self._keep_unpulled = keep_unpulled

        self._inlet = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Estado del stream conectado
        self.connected = False
        self.stream_name = ""
        self.stream_type = ""
        self.source_id_val = ""
        self.n_channels = 0
        self.channel_names: List[str] = []
        self.nominal_srate = float(DEFAULT_FS)

        # Buffers
        self.buffer_size = int(DEFAULT_FS * self.buffer_secs)
        self._buffers: List[deque] = []
        self._ts: deque = deque()
        self._unpulled: deque = deque()

        # Grabación opcional en memoria (solo datos, sin GUI)
        self._recording = False
        self._recorded: List[list] = []

        # Medición de Fs real
        self._rate = {"count": 0, "t0": time.perf_counter(), "hz": 0.0, "spp": 0}

    # ── Conexión ──────────────────────────────────────────────────────────
    def connect(self, timeout: float = 5.0) -> bool:
        """Resuelve un stream que coincida con los filtros y se conecta.
        Devuelve True si lo logró."""
        if self.connected:
            return True
        desc = find_stream(self._name, self._stype, self._source_id, timeout)
        if desc is None:
            log.warning("No se encontró ningún stream que coincida "
                        "(name=%r, type=%r, source_id=%r).",
                        self._name, self._stype, self._source_id)
            return False
        return self.connect_to(desc)

    def connect_to(self, desc) -> bool:
        """Conecta a un StreamDescription concreto (p. ej. de list_streams())."""
        if self.connected:
            log.warning("Ya hay un stream conectado; ignorando connect_to().")
            return False

        info = desc._info if isinstance(desc, StreamDescription) else desc
        if info is None:
            log.error("Descriptor de stream inválido (sin StreamInfo).")
            return False

        try:
            inlet = StreamInlet(
                info, max_buflen=int(max(1, self.buffer_secs)),
                recover=self._recover,
                processing_flags=proc_clocksync | proc_dejitter)
            full = inlet.info()
        except Exception as e:
            log.error("No se pudo abrir el inlet: %s", e)
            return False

        n_ch = full.channel_count()
        srate = full.nominal_srate() or DEFAULT_FS
        names = self._read_channel_labels(full, n_ch)
        n_use = n_ch if self._max_channels is None else min(n_ch, self._max_channels)

        self.buffer_size = int(max(1, srate) * self.buffer_secs)
        with self._lock:
            self._inlet = inlet
            self.n_channels = n_use
            self.channel_names = names[:n_use]
            self.nominal_srate = float(srate)
            self.stream_name = full.name()
            self.stream_type = full.type()
            self.source_id_val = full.source_id()
            self._buffers = [deque(maxlen=self.buffer_size) for _ in range(n_use)]
            self._ts = deque(maxlen=self.buffer_size)
            self._unpulled = deque(maxlen=self.buffer_size)
            self._recorded = []
            self._recording = False
            self._rate = {"count": 0, "t0": time.perf_counter(),
                          "hz": 0.0, "spp": 0}
            self.connected = True

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True,
                                        name="LSL-reader")
        self._thread.start()

        if n_ch > n_use:
            log.info("Conectado a '%s' (%d canales, usando %d) @ %.0f Hz",
                     self.stream_name, n_ch, n_use, srate)
        else:
            log.info("Conectado a '%s' (%d canales @ %.0f Hz)",
                     self.stream_name, n_use, srate)
        return True

    def disconnect(self):
        """Detiene la lectura y cierra el inlet."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._inlet is not None:
                try:
                    self._inlet.close_stream()
                except Exception:
                    pass
            self._inlet = None
            self.connected = False
            self.n_channels = 0
            self.channel_names = []
            for b in self._buffers:
                b.clear()
            self._ts.clear()
            self._unpulled.clear()
        log.info("Desconectado.")

    stop = disconnect  # alias

    # ── Hilo de lectura ───────────────────────────────────────────────────
    def _read_loop(self):
        inlet = self._inlet
        while self._running and inlet is not None:
            try:
                samples, timestamps = inlet.pull_chunk(timeout=0.1,
                                                       max_samples=1024)
            except Exception as e:
                log.error("Error en pull_chunk: %s", e)
                break
            if not samples:
                continue

            n_new = len(samples)
            nch = self.n_channels
            with self._lock:
                for row in samples:
                    for ch in range(nch):
                        self._buffers[ch].append(float(row[ch]))
                for ts in timestamps:
                    self._ts.append(ts)
                if self._keep_unpulled:
                    for ts, row in zip(timestamps, samples):
                        self._unpulled.append(
                            (ts, [float(row[ch]) for ch in range(nch)]))
                if self._recording:
                    for ts, row in zip(timestamps, samples):
                        self._recorded.append(
                            [ts] + [float(row[ch]) for ch in range(nch)])

            self._update_rate(n_new)

            if self._on_data is not None:
                try:
                    self._on_data(samples, timestamps)
                except Exception:
                    log.exception("Error en el callback on_data")

    def _update_rate(self, n_new: int):
        now = time.perf_counter()
        r = self._rate
        r["count"] += n_new
        r["spp"] = n_new
        elapsed = now - r["t0"]
        if elapsed >= 1.0:
            r["hz"] = r["count"] / elapsed
            r["count"] = 0
            r["t0"] = now

    # ── Metadata ──────────────────────────────────────────────────────────
    @staticmethod
    def _read_channel_labels(info, n_ch: int) -> List[str]:
        """Nombres de canal desde el XML desc(); respaldo EMG_i."""
        labels: List[str] = []
        try:
            ch = info.desc().child("channels").child("channel")
            while not ch.empty():
                lbl = ch.child_value("label")
                labels.append(lbl if lbl else f"EMG_{len(labels)+1}")
                ch = ch.next_sibling()
        except Exception:
            pass
        if len(labels) < n_ch:
            labels += [f"EMG_{i+1}" for i in range(len(labels), n_ch)]
        return labels

    @property
    def info(self) -> StreamInfoData:
        with self._lock:
            return StreamInfoData(
                name=self.stream_name,
                type=self.stream_type,
                source_id=self.source_id_val,
                n_channels=self.n_channels,
                nominal_srate=self.nominal_srate,
                channel_names=list(self.channel_names),
            )

    def channel_index(self, name: str) -> int:
        """Índice del canal por nombre (-1 si no existe)."""
        with self._lock:
            try:
                return self.channel_names.index(name)
            except ValueError:
                return -1

    def measured_rate(self) -> float:
        """Fs real medida (Hz) durante el último segundo."""
        return self._rate.get("hz", 0.0)

    def samples_per_chunk(self) -> int:
        return self._rate.get("spp", 0)

    def is_connected(self) -> bool:
        return self.connected

    # ── Acceso a datos: buffer circular (polling) ─────────────────────────
    def get_channel(self, channel: int) -> list:
        """Buffer completo (lista de floats) de un canal."""
        with self._lock:
            if 0 <= channel < self.n_channels:
                return list(self._buffers[channel])
        return []

    # alias por compatibilidad con el código previo
    get_data = get_channel

    def snapshot(self, as_array: Optional[bool] = None):
        """Todo el buffer, alineado. Devuelve (times, data):
        data tiene forma (muestras, canales) —np.ndarray si hay numpy y
        as_array no es False; si no, lista de filas."""
        as_array = HAS_NUMPY if as_array is None else as_array
        with self._lock:
            cols = [list(b) for b in self._buffers]
            times = list(self._ts)
            nch = self.n_channels
        m = min([len(times)] + [len(c) for c in cols]) if cols else 0
        times = times[-m:]
        cols = [c[-m:] for c in cols]

        if as_array and HAS_NUMPY:
            if m == 0:
                return np.empty(0), np.empty((0, nch))
            return np.asarray(times), np.asarray(cols, dtype=float).T
        rows = [[cols[ch][i] for ch in range(len(cols))] for i in range(m)]
        return times, rows

    def get_latest(self, n: int, as_array: Optional[bool] = None):
        """Como snapshot() pero solo las últimas n muestras."""
        times, data = self.snapshot(as_array=as_array)
        if HAS_NUMPY and isinstance(data, np.ndarray):
            return times[-n:], data[-n:]
        return times[-n:], data[-n:]

    # ── Acceso a datos: solo lo nuevo (pull) ──────────────────────────────
    def pull(self, as_array: Optional[bool] = None):
        """Devuelve (times, data) de las muestras nuevas desde la última
        llamada y vacía la cola. Requiere keep_unpulled=True (por defecto)."""
        as_array = HAS_NUMPY if as_array is None else as_array
        with self._lock:
            items = list(self._unpulled)
            self._unpulled.clear()
            nch = self.n_channels
        times = [t for t, _ in items]
        rows = [r for _, r in items]
        if as_array and HAS_NUMPY:
            if not rows:
                return np.empty(0), np.empty((0, nch))
            return np.asarray(times), np.asarray(rows, dtype=float)
        return times, rows

    # ── Utilidades ────────────────────────────────────────────────────────
    def wait_until_ready(self, min_samples: int = 1, timeout: float = 5.0) -> bool:
        """Bloquea hasta que hayan llegado al menos min_samples (o timeout)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                ready = self.n_channels > 0 and len(self._ts) >= min_samples
            if ready:
                return True
            time.sleep(0.02)
        return False

    # ── Grabación en memoria (opcional, solo datos) ───────────────────────
    def start_recording(self):
        """Empieza a acumular todas las muestras (con timestamp) en memoria."""
        with self._lock:
            self._recorded = []
            self._recording = True

    def stop_recording(self) -> list:
        """Detiene la grabación y devuelve las filas [ts, ch0, ch1, ...]."""
        with self._lock:
            self._recording = False
            return list(self._recorded)

    def save_csv(self, path: str, rows: Optional[Sequence[Sequence]] = None) -> bool:
        """Guarda a CSV las filas grabadas (o las que pases) con cabecera."""
        with self._lock:
            data = list(self._recorded) if rows is None else list(rows)
            names = list(self.channel_names[:self.n_channels])
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["lsl_time"] + (names or
                           [f"EMG_{i+1}" for i in range(len(data[0]) - 1)] if data else []))
                w.writerows(data)
            log.info("Guardado CSV -> %s (%d muestras)", path, len(data))
            return True
        except Exception as e:
            log.error("Error al guardar CSV: %s", e)
            return False

    # ── Context manager ───────────────────────────────────────────────────
    def __enter__(self):
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
        return False

    def __repr__(self):
        st = "conectado" if self.connected else "desconectado"
        return (f"<LSLStream {st} name={self.stream_name!r} "
                f"ch={self.n_channels} fs={self.nominal_srate:.0f}>")


# ═══════════════════════════════════════════════════════════════════════════
# Demo por consola (sin GUI): python lsl_stream.py [nombre_del_stream]
# ═══════════════════════════════════════════════════════════════════════════
def _demo(name: Optional[str] = None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    print("Buscando streams LSL…")
    streams = list_streams(timeout=2.0)
    if not streams:
        print("No se encontró ningún stream. ¿Está publicando el puente?")
        return
    for s in streams:
        print(f"  • {s.name}  [{s.type}]  {s.n_channels} ch  "
              f"{s.nominal_srate:.0f} Hz  (source_id={s.source_id})")

    emg = LSLStream(name=name)          # si name=None, toma el primero que haya
    if name is None:
        if not emg.connect_to(streams[0]):
            return
    elif not emg.connect(timeout=5.0):
        return

    try:
        emg.wait_until_ready(min_samples=1, timeout=5.0)
        print(f"\nConectado: {emg.info.name}  "
              f"{emg.info.n_channels} canales  "
              f"canales={emg.info.channel_names}\n")
        for _ in range(10):
            time.sleep(2.0)
            times, data = emg.get_latest(100)
            shape = data.shape if HAS_NUMPY and hasattr(data, "shape") \
                else f"{len(data)}x{emg.info.n_channels}"
            print(f"Fs real: {emg.measured_rate():6.1f} Hz  |  "
                  f"últimas 5 muestras -> {shape}")
    except KeyboardInterrupt:
        pass
    finally:
        emg.disconnect()


if __name__ == "__main__":
    import sys
    _demo(sys.argv[1] if len(sys.argv) > 1 else None)

def send_output_to_socket(stop_event, output_queue):
    while not stop_event.is_set(): 
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  
                s.connect(('localhost', 8052))
                print("Connected to Visualizer successfully.")

                while not stop_event.is_set():
                    try:
                        output_value = output_queue.get(timeout=1)
                        s.sendall(int(output_value).to_bytes(4, byteorder='little'))
                    except Empty:
                        continue  
                    except BrokenPipeError:
                        print("Connection lost. Reconnecting...")
                        break  

        except ConnectionRefusedError:
            print("Connection refused. Ensure the Visualizer is running. Retrying in 40 seconds...")
            time.sleep(40)
        except KeyboardInterrupt:
            print("\nSocket communication interrupted. Closing connection...")
            break 

    print("Socket thread stopped.")
            