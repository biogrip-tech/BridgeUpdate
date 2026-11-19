#!/usr/bin/env python3
"""
udp_to_lsl_bridge.py — Puente UDP -> LSL v2 para el sistema EMG (ESP32), Luckfox.

Novedades respecto a v1:
  - Varios dispositivos (ESP32) a la vez: cada uno se identifica por su IP de
    origen y/o por el puerto UDP al que envía, y publica sus propios streams.
  - Un hilo de recepción por puerto, separado del procesamiento: el socket se
    vacía siempre, aunque el filtrado de un dispositivo vaya lento.
  - Stream crudo ("<nombre>_raw") y procesado ("<nombre>") por dispositivo.
  - Pipeline configurable por etapas y por canal: IIR, notch, CAR, whitening
    y plugins de Python (ver pipeline.py).
  - Timestamps a partir del contador del paquete (menos jitter que la hora
    de llegada) y relleno con NaN de los paquetes perdidos.
  - Recarga en caliente: si cambia la config o un plugin, se aplica sin
    reiniciar el servicio. Cambiar solo el pipeline no toca el stream crudo.
  - Stream de estado (JSON) con tasa real, pérdidas y tiempos de cada etapa.
  - Compatible con el formato de config antiguo (el que escribe la GUI).

Formato del datagrama UDP del ESP32 (sendWindowUDP en el firmware):
    struct __attribute__((packed)) WindowHeader {
        uint32_t counter; uint16_t nChannels; uint16_t nSamples;
    };
    float window[nSamples * nChannels];   // float32, sample-major

Uso:
    python3 udp_to_lsl_bridge.py --config /etc/synaptic-bridge/bridge_config.json
    python3 udp_to_lsl_bridge.py --config cfg.json --check     # valida y mide el pipeline
    python3 udp_to_lsl_bridge.py --config cfg.json --check --json

Ver README_bridge_v2.md para el formato completo de la config.
"""

import argparse
import copy
import errno
import json
import logging
import os
import queue
import signal as os_signal
import socket
import struct
import sys
import threading
import time

import numpy as np
from pylsl import (IRREGULAR_RATE, StreamInfo, StreamOutlet, cf_float32,
                   cf_string, local_clock)

from pipeline import Pipeline, legacy_filters_to_pipeline

log = logging.getLogger("bridge")

HEADER_FMT = "<IHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # = 8 bytes
# Bit 15 de nChannels: las muestras vienen en int16 (little-endian) en vez de
# float32. Paquete la mitad de grande (ADC de 12 bits del Axon). Un puente
# antiguo ve >64 canales y descarta el paquete como inválido (no lee basura).
FLAG_INT16 = 0x8000
DEFAULT_CONFIG_PATH = os.path.expanduser("~/bridge_config.json")
HOSTNAME = socket.gethostname()

GLOBAL_DEFAULTS = {
    "host": "0.0.0.0",
    "rcvbuf_bytes": 4 * 1024 * 1024,   # búfer del socket (ver README: rmem_max)
    "queue_size": 2000,                # paquetes en espera por dispositivo
    "max_buffered_s": 30,              # segundos que LSL guarda por consumidor
    "stats_interval_s": 5.0,
    "stage_time_budget_ms": 5.0,       # aviso si una etapa tarda más que esto
    "status_stream": True,
    "status_stream_name": None,        # por defecto: SynapticBridge_status_<hostname>
    "plugins_dir": None,               # por defecto: <carpeta de la config>/plugins
    "data_dir": None,                  # por defecto: carpeta de la config
    "discovery_enabled": True,         # escribe un JSON con los orígenes UDP vistos
    "discovery_path": "/tmp/synaptic-bridge-discovery.json",
    "discovery_interval_s": 1.0,
    "discovery_stale_s": 120.0,
    # Un dispositivo "cualquier IP" (match {}) se queda con el PRIMER ESP32
    # que le envía. Paquetes de otra IP en ese puerto se descartan (y se
    # avisa) hasta que el primero lleve este tiempo sin enviar. Así dos ESP32
    # nunca se mezclan en el mismo stream.
    "catch_all_hold_s": 3.0,
    # Cada cuánto se buscan en la red otros streams LSL con el mismo nombre o
    # source_id que los nuestros (0 = no buscar).
    "lsl_conflict_check_s": 30.0,
}

DEVICE_DEFAULTS = {
    "name": None,
    "match": {},                       # {"ip": "192.168.11.27"} o vacío = cualquier IP
    "udp_port": 50043,
    "fs": 1000.0,
    "n_channels": 8,
    "stream_name": None,               # por defecto: BioMyo_EMG_<name>
    "stream_type": "EMG",
    "source_id": None,                 # por defecto: <hostname>-<name>
    "channel_labels": None,            # por defecto: EMG_1..EMG_N
    "unit": "microvolts",
    "publish_raw": True,
    "timestamps": "counter",           # "counter" o "arrival"
    "fill_gaps": True,                 # rellenar paquetes perdidos con NaN
    "max_gap_packets": 100,            # huecos mayores = reinicio del ESP32
    "pipeline": [],
}

# Claves globales que, si cambian, obligan a recrear los dispositivos.
WORKER_GLOBAL_KEYS = ("queue_size", "max_buffered_s", "plugins_dir", "data_dir",
                      "stage_time_budget_ms")


class ConfigError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Configuración
# ═══════════════════════════════════════════════════════════════════════════
def _clean(d):
    """Quita claves de comentario ("_comment", "_nota"...)."""
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "si", "sí")
    return bool(value)


def load_config(path, legacy_overrides=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ConfigError(f"no existe {path}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} no es JSON válido: {e}")
    if not isinstance(raw, dict):
        raise ConfigError("la raíz de la config debe ser un objeto JSON")
    return normalize_config(raw, path, legacy_overrides)


def normalize_config(raw, path, legacy_overrides=None):
    base_dir = os.path.dirname(os.path.abspath(path))
    g = dict(GLOBAL_DEFAULTS)

    if "devices" in raw:
        legacy = False
        g.update(_clean(raw.get("bridge") or {}))
        devs_raw = raw["devices"]
    else:
        # Formato v1 (un solo dispositivo, filtros fijos). Se mantiene el
        # comportamiento anterior: un único stream, sin "_raw".
        legacy = True
        if "host" in raw:
            g["host"] = raw["host"]
        d = {k: raw[k] for k in ("udp_port", "fs", "n_channels", "stream_name",
                                 "stream_type", "source_id") if k in raw}
        d.setdefault("stream_name", "BioMyo_EMG")
        d.setdefault("source_id", "esp32-emg-udp-bridge")
        d["name"] = "default"
        d["publish_raw"] = False
        d["pipeline"] = legacy_filters_to_pipeline(raw.get("filters") or {})
        if legacy_overrides:
            ov = {k: v for k, v in legacy_overrides.items() if v is not None}
            if "host" in ov:
                g["host"] = ov.pop("host")
            d.update(ov)
        devs_raw = [d]

    g["plugins_dir"] = g["plugins_dir"] or os.path.join(base_dir, "plugins")
    g["data_dir"] = g["data_dir"] or base_dir
    g["status_stream_name"] = g["status_stream_name"] or f"SynapticBridge_status_{HOSTNAME}"
    for key in ("rcvbuf_bytes", "queue_size"):
        g[key] = int(g[key])
    for key in ("max_buffered_s", "stats_interval_s", "stage_time_budget_ms",
                "discovery_interval_s", "discovery_stale_s", "catch_all_hold_s",
                "lsl_conflict_check_s"):
        g[key] = float(g[key])
    g["discovery_enabled"] = _as_bool(g["discovery_enabled"])
    g["discovery_path"] = str(g["discovery_path"] or "")
    g["discovery_interval_s"] = max(0.5, g["discovery_interval_s"])
    g["discovery_stale_s"] = max(5.0, g["discovery_stale_s"])
    g["catch_all_hold_s"] = max(0.5, g["catch_all_hold_s"])
    g["lsl_conflict_check_s"] = max(0.0, g["lsl_conflict_check_s"])

    if not isinstance(devs_raw, list) or not devs_raw:
        raise ConfigError("'devices' debe ser una lista con al menos un dispositivo")

    devices = []
    for i, dr in enumerate(devs_raw):
        if not isinstance(dr, dict):
            raise ConfigError(f"dispositivo #{i + 1}: debe ser un objeto JSON")
        d = copy.deepcopy(DEVICE_DEFAULTS)
        d.update(_clean(dr))
        if not d.get("name"):
            raise ConfigError(f"dispositivo #{i + 1}: falta 'name'")
        name = d["name"] = str(d["name"])
        d["stream_name"] = str(d.get("stream_name") or f"BioMyo_EMG_{name}")
        d["source_id"] = str(d.get("source_id") or f"{HOSTNAME}-{name}")
        try:
            d["udp_port"] = int(d["udp_port"])
            d["fs"] = float(d["fs"])
            d["n_channels"] = int(d["n_channels"])
            d["max_gap_packets"] = int(d["max_gap_packets"])
        except (TypeError, ValueError) as e:
            raise ConfigError(f"dispositivo '{name}': valor numérico inválido ({e})")
        if not 1 <= d["udp_port"] <= 65535:
            raise ConfigError(f"dispositivo '{name}': udp_port fuera de rango")
        if d["fs"] <= 0:
            raise ConfigError(f"dispositivo '{name}': fs debe ser > 0")
        if not 1 <= d["n_channels"] <= 64:
            raise ConfigError(f"dispositivo '{name}': n_channels debe estar entre 1 y 64")
        if d["timestamps"] not in ("counter", "arrival"):
            raise ConfigError(f"dispositivo '{name}': timestamps debe ser 'counter' o 'arrival'")
        match = d.get("match") or {}
        if not isinstance(match, dict):
            raise ConfigError(f"dispositivo '{name}': 'match' debe ser un objeto")
        ip = match.get("ip")
        if ip:
            try:
                socket.inet_aton(ip)
            except OSError:
                raise ConfigError(f"dispositivo '{name}': IP inválida '{ip}'")
            d["match"] = {"ip": str(ip)}
        else:
            d["match"] = {}
        if not isinstance(d["pipeline"], list):
            raise ConfigError(f"dispositivo '{name}': 'pipeline' debe ser una lista")
        devices.append(d)

    # Unicidad de nombres, source_id y rutas (puerto + IP).
    names, sids, routes = set(), set(), set()
    for d in devices:
        if d["name"] in names:
            raise ConfigError(f"nombre de dispositivo repetido: '{d['name']}'")
        names.add(d["name"])
        for sid in (d["source_id"], d["source_id"] + "_raw"):
            if sid in sids:
                raise ConfigError(f"source_id repetido: '{sid}' (cada stream necesita uno único)")
            sids.add(sid)
        route = (d["udp_port"], d["match"].get("ip", "*"))
        if route in routes:
            who = "cualquier IP" if route[1] == "*" else route[1]
            raise ConfigError(f"dos dispositivos reciben del puerto {route[0]} desde {who}")
        routes.add(route)

    # Dos streams con el mismo nombre: un consumidor que busca por nombre
    # (LabRecorder, la GUI) puede conectarse a cualquiera de los dos y saltar
    # al otro al reconectarse. No se rechaza (rompería configs existentes),
    # pero se avisa en el log y en --check.
    warnings, seen = [], {}
    for d in devices:
        for sname in [d["stream_name"]] + ([d["stream_name"] + "_raw"] if d["publish_raw"] else []):
            if sname in seen and seen[sname] != d["name"]:
                warnings.append(f"los dispositivos '{seen[sname]}' y '{d['name']}' publican un "
                                f"stream con el mismo nombre '{sname}': ponles stream_name distintos")
            seen.setdefault(sname, d["name"])

    return {"bridge": g, "devices": devices, "legacy": legacy, "warnings": warnings}


def device_identity(dcfg, gcfg):
    """Todo lo que, si cambia, obliga a recrear el dispositivo (no el pipeline)."""
    ident = {k: v for k, v in dcfg.items() if k != "pipeline"}
    ident["_global"] = {k: gcfg[k] for k in WORKER_GLOBAL_KEYS}
    return json.dumps(ident, sort_keys=True)


# ═══════════════════════════════════════════════════════════════════════════
# Utilidades
# ═══════════════════════════════════════════════════════════════════════════
def decode_header(packet):
    """Devuelve (counter, n_canales, n_muestras, dtype) o None si no es válido.
    dtype es "<f4" (formato original) o "<i2" (bit FLAG_INT16 activo)."""
    if len(packet) < HEADER_SIZE:
        return None
    counter, n_ch, n_samp = struct.unpack_from(HEADER_FMT, packet, 0)
    dtype = "<f4"
    if n_ch & FLAG_INT16:
        n_ch &= ~FLAG_INT16 & 0xFFFF
        dtype = "<i2"
    if n_ch <= 0 or n_ch > 64 or n_samp <= 0 or n_samp > 100000:
        return None
    return counter, n_ch, n_samp, dtype


def parse_packet(packet):
    """Devuelve (counter, ventana float32 (n_muestras, n_canales)) o None.
    Acepta muestras float32 o int16 (se convierten a float32 sin escalar)."""
    hdr = decode_header(packet)
    if hdr is None:
        return None
    counter, n_ch, n_samp, dtype = hdr
    itemsize = np.dtype(dtype).itemsize
    avail = (len(packet) - HEADER_SIZE) // itemsize
    use = min(n_samp * n_ch, avail)
    use -= use % n_ch
    if use <= 0:
        return None
    vals = np.frombuffer(packet, dtype=dtype, count=use, offset=HEADER_SIZE)
    if dtype != "<f4":
        vals = vals.astype(np.float32)
    return counter, vals.reshape(use // n_ch, n_ch)


def push_chunk(outlet, data, ts):
    """Publica un bloque (n_muestras, n_canales). Usa numpy directamente si la
    versión de pylsl lo admite (mucho más rápido que .tolist())."""
    arr = np.ascontiguousarray(data, dtype=np.float32)
    try:
        outlet.push_chunk(arr, ts)
    except (TypeError, ValueError):
        outlet.push_chunk(arr.tolist(), ts)


def udp_socket_drops(port):
    """Suma la columna 'drops' de /proc/net/udp para un puerto local (Linux).
    Devuelve None si no se puede leer (otro SO)."""
    total, found = 0, False
    for fn in ("/proc/net/udp", "/proc/net/udp6"):
        try:
            with open(fn, "r") as f:
                next(f, None)
                for line in f:
                    parts = line.split()
                    if len(parts) < 13:
                        continue
                    try:
                        if int(parts[1].rsplit(":", 1)[1], 16) == port:
                            total += int(parts[-1])
                            found = True
                    except ValueError:
                        continue
        except OSError:
            continue
    return total if found else None


class CounterClock:
    """Timestamps LSL a partir del índice de muestra, anclados a la llegada.

    Cada paquete llega con un retraso variable (Wi-Fi). La hora de llegada es
    siempre >= la hora real, así que se sigue la envolvente INFERIOR de las
    llegadas: se corrige rápido si un paquete llega antes de lo previsto y
    muy despacio si llega tarde. Así el jitter apenas afecta y se absorbe la
    deriva lenta entre el reloj del ESP32 y el del Luckfox.
    """

    FAST, SLOW, RESYNC_S = 0.1, 0.002, 1.0

    def __init__(self, fs):
        self.fs = float(fs)
        self.base = None

    def reset(self):
        self.base = None

    def predict(self, sample_index):
        return self.base + sample_index / self.fs

    def stamp(self, sample_index, t_arrival):
        """Devuelve el timestamp de la muestra `sample_index` (llegada t_arrival)."""
        pred = sample_index / self.fs
        if self.base is None:
            self.base = t_arrival - pred
            return t_arrival
        err = t_arrival - (self.base + pred)
        if abs(err) > self.RESYNC_S:
            # Demasiado lejos (fs mal configurada, pausa larga...): re-anclar.
            self.base = t_arrival - pred
            return t_arrival
        self.base += (self.FAST if err < 0 else self.SLOW) * err
        return self.base + pred


# ═══════════════════════════════════════════════════════════════════════════
# Dispositivo: procesa y publica los paquetes de un ESP32
# ═══════════════════════════════════════════════════════════════════════════
class DeviceWorker(threading.Thread):
    def __init__(self, dcfg, gcfg):
        super().__init__(daemon=True, name=f"dev-{dcfg['name']}")
        self.cfg = copy.deepcopy(dcfg)
        self.g = gcfg
        self.name_ = dcfg["name"]
        self.q = queue.Queue(maxsize=int(gcfg["queue_size"]))
        self._stop_ev = threading.Event()
        self._pending_pipeline = None
        self._lock = threading.Lock()

        self.pipeline = None
        self.raw_outlet = None
        self.proc_outlet = None
        self.n_channels = None
        self.clock = CounterClock(dcfg["fs"])
        self.last_counter = None
        self.sample_idx = 0
        self.last_packet_wall = None
        self.last_sender = None
        self.pipeline_errors = []
        self.prev_interval_had_data = False
        self._reset_counters()
        self.totals = {"packets": 0, "dropped": 0}

    # ── API usada desde otros hilos ───────────────────────────────────────
    def submit(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.c_overflow += 1

    def request_pipeline(self, specs):
        self._pending_pipeline = copy.deepcopy(specs)

    def stop(self):
        self._stop_ev.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass

    def snapshot(self, interval_s):
        """Estadísticas del último intervalo (y las reinicia)."""
        with self._lock:
            c = {
                "packets": self.c_packets, "samples": self.c_samples,
                "dropped": self.c_dropped, "duplicates": self.c_dup,
                "bad_packets": self.c_bad, "queue_overflow": self.c_overflow,
                "resyncs": self.c_resync, "channel_mismatch": self.c_chmismatch,
            }
            self._reset_counters()
        lost = c["dropped"]
        c["rate_sps"] = c["samples"] / interval_s if interval_s > 0 else 0.0
        c["loss_pct"] = 100.0 * lost / (c["packets"] + lost) if (c["packets"] + lost) else 0.0
        c["last_packet_age_s"] = (time.monotonic() - self.last_packet_wall
                                  if self.last_packet_wall else None)
        c["sender"] = self.last_sender
        c["pipeline"] = self.pipeline.stats() if self.pipeline else []
        c["pipeline_errors"] = list(self.pipeline_errors)
        c["queue_len"] = self.q.qsize()
        return c

    def watched_files(self):
        p = self.pipeline
        return p.watched_files() if p else {}

    # ── Hilo de trabajo ───────────────────────────────────────────────────
    def run(self):
        while not self._stop_ev.is_set():
            if self._pending_pipeline is not None:
                self._swap_pipeline()
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self._handle_packet(*item)
            except Exception:
                log.exception("[%s] error procesando un paquete", self.name_)
        self.raw_outlet = None
        self.proc_outlet = None

    def _reset_counters(self):
        self.c_packets = self.c_samples = self.c_dropped = self.c_dup = 0
        self.c_bad = self.c_overflow = self.c_resync = self.c_chmismatch = 0

    def _ctx(self):
        return {"plugins_dir": self.g["plugins_dir"], "data_dir": self.g["data_dir"]}

    def _build_pipeline(self, specs):
        p = Pipeline(self.cfg["fs"], self.n_channels, specs, self._ctx())
        self.pipeline_errors = list(p.errors)
        for err in p.errors:
            log.warning("[%s] etapa no aplicada: %s", self.name_, err)
        if p.stages:
            log.info("[%s] pipeline: %s", self.name_, " -> ".join(p.describe()))
        else:
            log.info("[%s] pipeline vacío: el stream procesado es igual al crudo", self.name_)
        return p

    def _make_outlet(self, raw):
        c = self.cfg
        suffix = "_raw" if raw else ""
        info = StreamInfo(name=c["stream_name"] + suffix, type=c["stream_type"],
                          channel_count=self.n_channels, nominal_srate=c["fs"],
                          channel_format=cf_float32, source_id=c["source_id"] + suffix)
        labels = c.get("channel_labels")
        if not labels or len(labels) != self.n_channels:
            labels = [f"EMG_{i + 1}" for i in range(self.n_channels)]
        chns = info.desc().append_child("channels")
        for label in labels:
            ch = chns.append_child("channel")
            ch.append_child_value("label", str(label))
            ch.append_child_value("unit", str(c.get("unit", "microvolts")))
            ch.append_child_value("type", c["stream_type"])
        info.desc().append_child_value("manufacturer", "BioGrip")
        acq = info.desc().append_child("acquisition")
        acq.append_child_value("device", "ESP32 EMGSystem")
        acq.append_child_value("transport", "UDP->LSL bridge v2 (Luckfox)")
        acq.append_child_value("bridge_host", HOSTNAME)
        acq.append_child_value("bridge_device", self.name_)
        acq.append_child_value("kind", "raw" if raw else "processed")
        chain = "none (raw)" if raw or not self.pipeline.describe() else ",".join(self.pipeline.describe())
        acq.append_child_value("filters_applied", chain)
        acq.append_child_value("pipeline", "[]" if raw else json.dumps(self.cfg["pipeline"]))
        max_buf = max(1, int(round(self.g["max_buffered_s"])))
        return StreamOutlet(info, chunk_size=0, max_buffered=max_buf)

    def _create_outlets(self, n_ch):
        self.n_channels = n_ch
        if n_ch != self.cfg["n_channels"]:
            log.warning("[%s] el ESP32 envía %d canales (la config dice %d); se usan %d",
                        self.name_, n_ch, self.cfg["n_channels"], n_ch)
        self.pipeline = self._build_pipeline(self.cfg["pipeline"])
        if self.cfg["publish_raw"]:
            self.raw_outlet = self._make_outlet(raw=True)
        self.proc_outlet = self._make_outlet(raw=False)
        names = [self.cfg["stream_name"]] + ([self.cfg["stream_name"] + "_raw"] if self.raw_outlet else [])
        log.info("[%s] streams LSL creados: %s (%d canales, %g Hz)",
                 self.name_, ", ".join(names), n_ch, self.cfg["fs"])

    def _swap_pipeline(self):
        specs, self._pending_pipeline = self._pending_pipeline, None
        self.cfg["pipeline"] = specs
        if self.n_channels is None:
            return      # todavía no llegó ningún paquete: se usará al crear
        self.pipeline = self._build_pipeline(specs)
        # Se recrea SOLO el stream procesado (sus metadatos describen el
        # pipeline). El crudo sigue intacto, así una grabación no se corta.
        # Los consumidores se reconectan solos: el source_id no cambia.
        self.proc_outlet = None
        self.proc_outlet = self._make_outlet(raw=False)
        log.info("[%s] pipeline recargado sin reiniciar el servicio", self.name_)

    def _push_both(self, raw, processed, ts):
        if self.raw_outlet is not None:
            push_chunk(self.raw_outlet, raw, ts)
        push_chunk(self.proc_outlet, processed, ts)

    def _handle_packet(self, t_arrival, sender_ip, packet):
        parsed = parse_packet(packet)
        if parsed is None:
            with self._lock:
                self.c_bad += 1
            return
        counter, win = parsed
        n_samp, n_ch = win.shape
        self.last_packet_wall = time.monotonic()
        self.last_sender = sender_ip

        if self.proc_outlet is None:
            self._create_outlets(n_ch)
        elif n_ch != self.n_channels:
            with self._lock:
                self.c_chmismatch += 1
            return

        # ── Contador: duplicados, pérdidas y reinicios ───────────────────
        missing_samples = 0
        if self.last_counter is not None:
            gap = (counter - self.last_counter) & 0xFFFFFFFF
            if gap == 0:
                with self._lock:
                    self.c_dup += 1
                return
            if gap >= 0x80000000 or gap > self.cfg["max_gap_packets"]:
                # Contador hacia atrás o salto enorme: el ESP32 se reinició o
                # hubo una pausa larga. Se re-ancla la línea de tiempo.
                log.info("[%s] salto de contador %d -> %d: re-sincronizando",
                         self.name_, self.last_counter, counter)
                self.clock.reset()
                self.sample_idx = 0
                with self._lock:
                    self.c_resync += 1
            elif gap > 1:
                lost = gap - 1
                missing_samples = lost * n_samp
                with self._lock:
                    self.c_dropped += lost
                self.totals["dropped"] += lost
        self.last_counter = counter

        counter_ts = self.cfg["timestamps"] == "counter"

        # Relleno de huecos con NaN: no pasa por el pipeline (un NaN dentro
        # de un IIR contaminaría su estado para siempre).
        if missing_samples and self.cfg["fill_gaps"] and self.clock.base is not None and counter_ts:
            nan_block = np.full((missing_samples, n_ch), np.nan, dtype=np.float32)
            ts_gap = self.clock.predict(self.sample_idx + missing_samples - 1)
            self._push_both(nan_block, nan_block, ts_gap)
        self.sample_idx += missing_samples

        idx_last = self.sample_idx + n_samp - 1
        ts = self.clock.stamp(idx_last, t_arrival) if counter_ts else t_arrival
        self.sample_idx += n_samp

        processed = self.pipeline.process(win)
        self._push_both(win, processed, ts)

        with self._lock:
            self.c_packets += 1
            self.c_samples += n_samp
        self.totals["packets"] += 1


# ═══════════════════════════════════════════════════════════════════════════
# Descubrimiento: orígenes UDP vistos, aunque no estén en la config
# ═══════════════════════════════════════════════════════════════════════════
class DiscoveryRegistry:
    """Registra los orígenes UDP sin publicar streams ni tocar la config.

    La GUI lee el JSON resultante para emparejar hardware nuevo. Se guarda una
    entrada por (IP, puerto), así que un segundo equipo en un puerto que ya
    tiene un dispositivo catch-all aparece como origen separado.
    """

    def __init__(self, stale_s=120.0, max_sources=256):
        self.stale_s = float(stale_s)
        self.max_sources = int(max_sources)
        self._lock = threading.Lock()
        self._sources = {}
        self._devices = {}

    def observe(self, ip, port, packet, t_seen, device_name=None):
        hdr = decode_header(packet)
        if hdr is None:
            return
        counter, n_ch, n_samp, dtype = hdr
        key = (str(ip), int(port))
        with self._lock:
            rec = self._sources.get(key)
            if rec is None:
                self._prune_locked(t_seen)
                if len(self._sources) >= self.max_sources:
                    return
                rec = {
                    "ip": key[0], "port": key[1], "packets": 0, "samples": 0,
                    "channels": n_ch, "samples_per_packet": n_samp,
                    "format": "int16" if dtype == "<i2" else "float32",
                    "first_seen": t_seen, "last_seen": t_seen,
                    "device": None, "rate_sps": 0.0,
                    "lost_packets": 0, "counter_resets": 0,
                    "_win_start": t_seen, "_win_samples": 0,
                    "_last_counter": None, "_int_packets": 0, "_int_lost": 0,
                }
                self._sources[key] = rec
            rec["packets"] += 1
            rec["_int_packets"] += 1
            rec["samples"] += n_samp
            # Pérdidas medidas en la recepción, por ESP32 (IP), antes de
            # cualquier cola o procesamiento: si aquí hay huecos, el paquete
            # no llegó al Luckfox (Wi-Fi / emisor), no es culpa de LSL.
            last = rec["_last_counter"]
            if last is not None:
                gap = (counter - last) & 0xFFFFFFFF
                if gap >= 0x80000000 or gap > 1000:
                    rec["counter_resets"] += 1
                elif gap > 1:
                    rec["lost_packets"] += gap - 1
                    rec["_int_lost"] += gap - 1
            rec["_last_counter"] = counter
            rec["channels"] = n_ch
            rec["samples_per_packet"] = n_samp
            rec["format"] = "int16" if dtype == "<i2" else "float32"
            rec["last_seen"] = t_seen
            rec["device"] = device_name or None
            if device_name:
                dev = self._devices.get(device_name)
                if dev is None:
                    dev = {"name": device_name, "port": key[1], "ips": {},
                           "last_seen": t_seen}
                    self._devices[device_name] = dev
                dev["port"] = key[1]
                dev["last_seen"] = t_seen
                seen = dev["ips"].get(key[0])
                if seen is None:
                    seen = {"ip": key[0], "first_seen": t_seen,
                            "last_seen": t_seen, "packets": 0}
                    dev["ips"][key[0]] = seen
                seen["last_seen"] = t_seen
                seen["packets"] += 1
            elapsed = t_seen - rec["_win_start"]
            rec["_win_samples"] += n_samp
            if elapsed >= 1.0:
                rec["rate_sps"] = rec["_win_samples"] / elapsed
                rec["_win_start"] = t_seen
                rec["_win_samples"] = 0
            elif elapsed >= 0.25:
                # Estimación provisional para ráfagas cortas de descubrimiento.
                rec["rate_sps"] = rec["_win_samples"] / elapsed

    def _prune_locked(self, now):
        cutoff = now - self.stale_s
        for key in [k for k, r in self._sources.items()
                    if r["last_seen"] < cutoff]:
            self._sources.pop(key, None)
        for name in list(self._devices):
            dev = self._devices[name]
            for ip in [i for i, r in dev["ips"].items()
                       if r["last_seen"] < cutoff]:
                dev["ips"].pop(ip, None)
            if not dev["ips"] or dev["last_seen"] < cutoff:
                self._devices.pop(name, None)

    def take_interval(self):
        """Paquetes y pérdidas por origen desde la última llamada (y reinicia)."""
        out = []
        with self._lock:
            for rec in self._sources.values():
                out.append({"ip": rec["ip"], "port": rec["port"], "device": rec["device"],
                            "packets": rec["_int_packets"], "lost": rec["_int_lost"]})
                rec["_int_packets"] = rec["_int_lost"] = 0
        return out

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            self._prune_locked(now)
            sources = []
            for rec in sorted(self._sources.values(),
                              key=lambda r: r["last_seen"], reverse=True):
                item = {k: v for k, v in rec.items()
                        if not k.startswith("_") and k not in
                        ("first_seen", "last_seen")}
                item["age_s"] = round(max(0.0, now - rec["last_seen"]), 1)
                item["seen_for_s"] = round(max(0.0, now - rec["first_seen"]), 1)
                sources.append(item)
            devices = []
            for dev in sorted(self._devices.values(), key=lambda d: d["name"]):
                ips = []
                for rec in sorted(dev["ips"].values(),
                                  key=lambda r: r["first_seen"]):
                    ips.append({
                        "ip": rec["ip"],
                        "packets": rec["packets"],
                        "age_s": round(max(0.0, now - rec["last_seen"]), 1),
                        "seen_for_s": round(max(0.0, now - rec["first_seen"]), 1),
                    })
                devices.append({
                    "name": dev["name"], "port": dev["port"], "ips": ips,
                    "age_s": round(max(0.0, now - dev["last_seen"]), 1),
                })
            return {"sources": sources, "devices": devices}


# ═══════════════════════════════════════════════════════════════════════════
# Recepción: un hilo por puerto UDP, solo lee y reparte
# ═══════════════════════════════════════════════════════════════════════════
class Receiver(threading.Thread):
    UNKNOWN_LOG_EVERY_S = 30.0

    def __init__(self, host, port, rcvbuf, registry=None, catch_all_hold_s=3.0):
        super().__init__(daemon=True, name=f"rx-{port}")
        self.host, self.port = host, port
        self._stop_ev = threading.Event()
        self.routes = ({}, None)       # (ip -> worker, worker para cualquier IP)
        self.registry = registry
        self.catch_all_hold_s = float(catch_all_hold_s)
        self._unknown_last = {}
        self._foreign_last = {}
        self.unknown_packets = 0
        self.foreign_packets = 0       # rechazados por el bloqueo del catch-all
        self._ca_worker = None         # catch-all al que pertenece el bloqueo
        self._ca_ip = None             # IP a la que está "enganchado" el catch-all
        self._ca_last = 0.0

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # SIN SO_REUSEADDR a propósito: con él, una segunda instancia del
        # puente (el servicio + una ejecución a mano, o un proceso colgado)
        # podía abrir el mismo puerto sin error y "robar" los paquetes en
        # silencio, dejando un stream LSL vivo pero sin datos.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(rcvbuf))
        except OSError:
            pass
        try:
            sock.bind((host, port))
        except OSError as e:
            sock.close()
            if e.errno == errno.EADDRINUSE:
                raise OSError(e.errno, f"el puerto UDP {port} ya está en uso por otro proceso "
                                       f"(¿otra instancia del puente? mira: sudo ss -ulpn | grep {port})")
            raise
        sock.settimeout(0.5)
        self.sock = sock
        actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        # Linux reporta el doble del valor efectivo pedido; si se queda corto
        # es porque net.core.rmem_max lo limita.
        if actual < rcvbuf:
            log.warning("puerto %d: búfer de recepción de %d KB (se pidieron %d KB). "
                        "Sube net.core.rmem_max (ver README) para evitar pérdidas.",
                        port, actual // 1024, rcvbuf // 1024)
        self._drops0 = udp_socket_drops(port)
        self._drops_prev = self._drops0
        log.info("escuchando UDP en %s:%d", host, port)

    def set_routes(self, by_ip, catch_all):
        self.routes = (dict(by_ip), catch_all)    # asignación atómica

    def stop(self):
        self._stop_ev.set()

    def kernel_drops_delta(self):
        """Datagramas que el kernel descartó en este puerto desde la última
        llamada (búfer del socket lleno = el puente no lee a tiempo)."""
        cur = udp_socket_drops(self.port)
        if cur is None or self._drops_prev is None:
            return None
        delta, self._drops_prev = max(0, cur - self._drops_prev), cur
        return delta

    def _route(self, ip, now):
        by_ip, catch_all = self.routes
        worker = by_ip.get(ip)
        if worker is not None or catch_all is None:
            return worker, False
        if catch_all is not self._ca_worker:      # cambió la config
            self._ca_worker, self._ca_ip = catch_all, None
        if self._ca_ip is None or self._ca_ip == ip:
            if self._ca_ip is None:
                log.info("[%s] catch-all del puerto %d enganchado a %s",
                         catch_all.name_, self.port, ip)
            self._ca_ip, self._ca_last = ip, now
            return catch_all, False
        if now - self._ca_last > self.catch_all_hold_s:
            log.warning("[%s] %s dejó de enviar hace %.1f s: el catch-all del puerto %d "
                        "pasa a %s", catch_all.name_, self._ca_ip, now - self._ca_last,
                        self.port, ip)
            self._ca_ip, self._ca_last = ip, now
            return catch_all, False
        return None, True        # otro ESP32 en un puerto catch-all ya ocupado

    def run(self):
        while not self._stop_ev.is_set():
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as e:
                if self._stop_ev.is_set():
                    break
                log.error("puerto %d: error de recepción: %s", self.port, e)
                time.sleep(0.1)
                continue
            t_arrival = local_clock()
            now = time.monotonic()
            ip = addr[0]
            worker, foreign = self._route(ip, now)
            if self.registry is not None:
                self.registry.observe(ip, self.port, data, now,
                                      worker.name_ if worker is not None else None)
            if worker is None:
                if foreign:
                    self._log_foreign(ip, now)
                else:
                    self._log_unknown(ip)
                continue
            worker.submit((t_arrival, ip, data))
        try:
            self.sock.close()
        except OSError:
            pass

    def _log_foreign(self, ip, now):
        self.foreign_packets += 1
        if now - self._foreign_last.get(ip, -1e9) >= self.UNKNOWN_LOG_EVERY_S:
            self._foreign_last[ip] = now
            name = self._ca_worker.name_ if self._ca_worker else "?"
            log.warning("puerto %d: %s también está enviando, pero el dispositivo '%s' "
                        "(match vacío) ya recibe de %s. Se DESCARTAN sus paquetes para no "
                        "mezclar dos ESP32 en un stream: añade un dispositivo con "
                        "\"match\": {\"ip\": \"%s\"} o usa otro puerto.",
                        self.port, ip, name, self._ca_ip, ip)

    def _log_unknown(self, ip):
        self.unknown_packets += 1
        now = time.monotonic()
        if now - self._unknown_last.get(ip, -1e9) >= self.UNKNOWN_LOG_EVERY_S:
            self._unknown_last[ip] = now
            log.warning("puerto %d: paquetes de %s sin dispositivo asignado en la config "
                        "(se descartan)", self.port, ip)

# ═══════════════════════════════════════════════════════════════════════════
# Puente: crea/actualiza dispositivos y receptores, vigila la config
# ═══════════════════════════════════════════════════════════════════════════
class Bridge:
    def __init__(self, config_path, legacy_overrides=None):
        self.config_path = config_path
        self.legacy_overrides = legacy_overrides
        self.workers = {}        # nombre -> DeviceWorker
        self.receivers = {}      # puerto -> Receiver
        self.cfg = None
        self.stop_ev = threading.Event()
        self.status_outlet = None
        self.discovery = DiscoveryRegistry()
        self._cfg_mtime = None
        self._watched = {}       # nombre -> {archivo: mtime}
        self._conflicts = {}     # uid del stream ajeno -> descripción
        self._conflict_thread = None
        self._conflict_misses = {}
        self._next_conflict_check = time.monotonic() + 3.0

    def start(self):
        cfg = load_config(self.config_path, self.legacy_overrides)
        self._cfg_mtime = self._mtime(self.config_path)
        self.apply(cfg, initial=True)

    @staticmethod
    def _mtime(path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def apply(self, cfg, initial=False):
        g = cfg["bridge"]
        self.discovery.stale_s = float(g["discovery_stale_s"])
        old_g = self.cfg["bridge"] if self.cfg else None
        new_devs = {d["name"]: d for d in cfg["devices"]}

        # 1. Dispositivos eliminados
        for name in list(self.workers):
            if name not in new_devs:
                log.info("[%s] eliminado de la config: se cierran sus streams", name)
                self._stop_worker(name)

        # 2. Nuevos o modificados
        for name, d in new_devs.items():
            w = self.workers.get(name)
            if w is not None and device_identity(w.cfg, w.g) == device_identity(d, g):
                if w.cfg["pipeline"] != d["pipeline"]:
                    w.request_pipeline(d["pipeline"])
                continue
            if w is not None:
                log.info("[%s] cambió la identidad del stream: se recrea", name)
                self._stop_worker(name)
            nw = DeviceWorker(d, g)
            nw.start()
            self.workers[name] = nw
            ip = d["match"].get("ip", "cualquier IP")
            log.info("[%s] esperando datos en el puerto %d desde %s -> stream '%s'",
                     name, d["udp_port"], ip, d["stream_name"])

        # 3. Receptores (uno por puerto)
        ports = {d["udp_port"] for d in cfg["devices"]}
        rebuild_all = old_g is not None and (old_g["host"] != g["host"] or
                                             old_g["rcvbuf_bytes"] != g["rcvbuf_bytes"])
        closing = []
        for port in list(self.receivers):
            if port not in ports or rebuild_all:
                r = self.receivers.pop(port)
                r.stop()
                closing.append(r)
        for r in closing:      # sin SO_REUSEADDR hay que esperar a que suelte el puerto
            r.join(timeout=2)
        for r in self.receivers.values():
            r.catch_all_hold_s = g["catch_all_hold_s"]
        for port in sorted(ports):
            if port in self.receivers:
                continue
            try:
                r = Receiver(g["host"], port, g["rcvbuf_bytes"], self.discovery,
                             g["catch_all_hold_s"])
            except OSError as e:
                msg = f"no se pudo abrir {g['host']}:{port}: {e}"
                if initial:
                    raise ConfigError(msg)
                log.error(msg)
                continue
            r.start()
            self.receivers[port] = r

        # 4. Rutas: qué IP de cada puerto va a qué dispositivo
        for port, r in self.receivers.items():
            by_ip, catch_all = {}, None
            for d in cfg["devices"]:
                if d["udp_port"] != port:
                    continue
                w = self.workers[d["name"]]
                if d["match"].get("ip"):
                    by_ip[d["match"]["ip"]] = w
                else:
                    catch_all = w
            r.set_routes(by_ip, catch_all)

        # 5. Stream de estado
        if g["status_stream"] and (self.status_outlet is None or
                                   (old_g and old_g["status_stream_name"] != g["status_stream_name"])):
            info = StreamInfo(name=g["status_stream_name"], type="Status", channel_count=1,
                              nominal_srate=IRREGULAR_RATE, channel_format=cf_string,
                              source_id=f"{HOSTNAME}-synaptic-bridge-status")
            self.status_outlet = StreamOutlet(info)
            log.info("stream de estado: %s", g["status_stream_name"])
        elif not g["status_stream"]:
            self.status_outlet = None

        for w in cfg.get("warnings", []):
            log.warning("config: %s", w)
        if cfg["legacy"] and initial:
            log.info("config en formato antiguo (un dispositivo); ver README_bridge_v2.md "
                     "para usar varios dispositivos y stream crudo")
        self.cfg = cfg

    def _stop_worker(self, name):
        w = self.workers.pop(name)
        w.stop()
        w.join(timeout=3)
        self._watched.pop(name, None)

    # ── Bucle principal: estadísticas y recarga ───────────────────────────
    def run(self):
        interval = self.cfg["bridge"]["stats_interval_s"]
        next_stats = time.monotonic() + interval
        last_stats = time.monotonic()
        disc_interval = self.cfg["bridge"]["discovery_interval_s"]
        next_disc = time.monotonic() + disc_interval
        while not self.stop_ev.wait(1.0):
            self._check_reload()
            now = time.monotonic()
            if now >= next_stats:
                self._report(now - last_stats)
                last_stats = now
                interval = self.cfg["bridge"]["stats_interval_s"]
                next_stats = now + interval
            cc = self.cfg["bridge"]["lsl_conflict_check_s"]
            if cc > 0 and now >= self._next_conflict_check and not (
                    self._conflict_thread and self._conflict_thread.is_alive()):
                self._next_conflict_check = now + cc
                self._conflict_thread = threading.Thread(
                    target=self._check_lsl_conflicts, daemon=True, name="lsl-conflicts")
                self._conflict_thread.start()
            if self.cfg["bridge"]["discovery_enabled"] and now >= next_disc:
                self._write_discovery()
                disc_interval = self.cfg["bridge"]["discovery_interval_s"]
                next_disc = now + disc_interval
        self.shutdown()

    def _check_reload(self):
        mtime = self._mtime(self.config_path)
        if mtime is not None and mtime != self._cfg_mtime:
            time.sleep(0.3)     # deja terminar una escritura en curso
            self._cfg_mtime = self._mtime(self.config_path)
            try:
                cfg = load_config(self.config_path, self.legacy_overrides)
            except ConfigError as e:
                log.error("config nueva inválida, se mantiene la anterior: %s", e)
                return
            log.info("config modificada: aplicando cambios")
            self.apply(cfg)
            return
        # Plugins o referencias .npz modificados -> reconstruir ese pipeline
        for name, w in self.workers.items():
            current = w.watched_files()
            prev = self._watched.get(name)
            if prev is not None and current and current != prev:
                log.info("[%s] cambió un plugin o referencia: recargando pipeline", name)
                w.request_pipeline(w.cfg["pipeline"])
            self._watched[name] = current

    def _own_streams(self):
        """[(nombre, source_id, uid)] de los outlets que publica este proceso."""
        own = []
        for w in list(self.workers.values()):
            for outlet in (w.raw_outlet, w.proc_outlet):
                if outlet is None:
                    continue
                try:
                    info = outlet.get_info()
                    own.append((info.name(), info.source_id(), info.uid()))
                except Exception:
                    pass
        return own

    def _check_lsl_conflicts(self):
        """Busca en la red streams LSL ajenos con el mismo nombre o source_id
        que los nuestros. Si existen, un consumidor (LabRecorder, la GUI...)
        puede conectarse al equivocado, o saltar a él al reconectarse, y se
        ve como que "la señal se cae"."""
        own = self._own_streams()
        if not own:
            return
        try:
            from pylsl import resolve_streams
            found = resolve_streams(wait_time=2.0)
        except Exception as e:
            log.debug("no se pudo buscar conflictos LSL: %s", e)
            return
        own_uids = {u for _, _, u in own}
        current = {}
        for info in found:
            uid = info.uid()
            if uid in own_uids:
                continue
            for name, sid, _ in own:
                if info.source_id() == sid or info.name() == name:
                    what = "source_id" if info.source_id() == sid else "nombre"
                    current[uid] = (f"'{info.name()}' (source_id={info.source_id()}) en "
                                    f"{info.hostname()} tiene el mismo {what} que nuestro "
                                    f"stream '{name}'")
                    break
        # Un resolve puede perder algún stream (multicast): un conflicto se da
        # por resuelto solo tras dos búsquedas seguidas sin verlo.
        misses = self._conflict_misses
        for uid in list(self._conflicts):
            if uid not in current:
                misses[uid] = misses.get(uid, 0) + 1
                if misses[uid] < 2:
                    current[uid] = self._conflicts[uid]
                    continue
                log.info("conflicto LSL resuelto: %s", self._conflicts[uid])
            misses.pop(uid, None)
        for uid, desc in current.items():
            if uid not in self._conflicts:
                log.warning("CONFLICTO LSL: %s. Los consumidores pueden conectarse al stream "
                            "equivocado. Usa stream_name/source_id únicos por Luckfox y "
                            "dispositivo, y asegúrate de que no haya otra instancia del "
                            "puente corriendo.", desc)
        self._conflicts = current

    def _write_discovery(self):
        path = self.cfg["bridge"].get("discovery_path")
        if not path:
            return
        payload = {"host": HOSTNAME, "time": time.time()}
        payload.update(self.discovery.snapshot())
        try:
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError as e:
            log.warning("no se pudo escribir el descubrimiento %s: %s", path, e)

    def _report(self, interval_s):
        g = self.cfg["bridge"]
        budget = g["stage_time_budget_ms"]
        status = {"host": HOSTNAME, "time": time.time(), "devices": {}}
        for name, w in self.workers.items():
            s = w.snapshot(interval_s)
            status["devices"][name] = s
            fs = w.cfg["fs"]
            had_data_before = w.prev_interval_had_data
            w.prev_interval_had_data = s["packets"] > 0
            if s["packets"] == 0:
                age = s["last_packet_age_s"]
                if age is None:
                    log.info("[%s] sin datos todavía (puerto %d)", name, w.cfg["udp_port"])
                else:
                    log.warning("[%s] sin datos desde hace %.0f s", name, age)
                continue
            log.info("[%s] %.0f muestras/s | paquetes=%d perdidos=%d (%.1f%%) | "
                     "cola_llena=%d inválidos=%d | origen=%s",
                     name, s["rate_sps"], s["packets"], s["dropped"], s["loss_pct"],
                     s["queue_overflow"], s["bad_packets"], s["sender"])
            # Solo con dos intervalos seguidos con datos (el primero y el
            # último de una sesión están incompletos y darían falsos avisos).
            if had_data_before and s["rate_sps"] > 0 and abs(s["rate_sps"] - fs) / fs > 0.10 and s["loss_pct"] < 5:
                log.warning("[%s] llegan ~%.0f muestras/s pero 'fs' es %g Hz: revisa 'fs' "
                            "(los filtros y timestamps dependen de ella)", name, s["rate_sps"], fs)
            for st in s["pipeline"]:
                if st["avg_ms"] > budget:
                    log.warning("[%s] la etapa %s tarda %.2f ms de media (límite %.1f ms)",
                                name, st["stage"], st["avg_ms"], budget)
        ports = {}
        for port, r in list(self.receivers.items()):
            drops = r.kernel_drops_delta()
            ports[port] = {"kernel_drops": drops, "foreign_packets": r.foreign_packets,
                           "unknown_packets": r.unknown_packets,
                           "catch_all_ip": r._ca_ip}
            if drops:
                log.warning("puerto %d: el kernel descartó %d datagramas (búfer del socket "
                            "lleno): el Luckfox no da abasto o el búfer es pequeño "
                            "(rcvbuf_bytes / net.core.rmem_max)", port, drops)
        sources = self.discovery.take_interval()
        for src in sources:
            if src["lost"]:
                total = src["packets"] + src["lost"]
                log.warning("origen %s:%d (%s): %d de %d paquetes NO llegaron al Luckfox "
                            "(%.1f%%) -> pérdida en Wi-Fi o en el ESP32, no en LSL",
                            src["ip"], src["port"], src["device"] or "sin asignar",
                            src["lost"], total, 100.0 * src["lost"] / total)
        status["ports"] = {str(k): v for k, v in ports.items()}
        status["sources"] = sources
        status["lsl_conflicts"] = list(self._conflicts.values())
        if self.status_outlet is not None:
            try:
                self.status_outlet.push_sample([json.dumps(status, default=str)])
            except Exception as e:
                log.debug("no se pudo publicar el estado: %s", e)

    def shutdown(self):
        for r in self.receivers.values():
            r.stop()
        for name in list(self.workers):
            self._stop_worker(name)
        for r in self.receivers.values():
            r.join(timeout=2)
        self.receivers.clear()
        self.status_outlet = None
        log.info("puente detenido")


# ═══════════════════════════════════════════════════════════════════════════
# --check: valida la config y mide el pipeline sin abrir sockets ni LSL
# ═══════════════════════════════════════════════════════════════════════════
def run_check(cfg, samples_per_window, n_windows, as_json):
    g = cfg["bridge"]
    ctx = {"plugins_dir": g["plugins_dir"], "data_dir": g["data_dir"]}
    report, ok = {"devices": {}}, True
    rng = np.random.default_rng(0)
    for d in cfg["devices"]:
        p = Pipeline(d["fs"], d["n_channels"], d["pipeline"], ctx)
        for _ in range(n_windows):
            p.process(rng.standard_normal((samples_per_window, d["n_channels"])) * 50.0)
        stages = p.stats()
        errors = list(p.errors) + [f"{s['stage']}: {s['error']}" for s in stages if s["disabled"]]
        ok = ok and not errors
        window_ms = 1000.0 * samples_per_window / d["fs"]
        total_ms = sum(s["avg_ms"] for s in stages if not s["disabled"])
        report["devices"][d["name"]] = {
            "stream_name": d["stream_name"], "fs": d["fs"], "n_channels": d["n_channels"],
            "window_ms": round(window_ms, 2), "pipeline_avg_ms": round(total_ms, 3),
            "stages": stages, "errors": errors,
        }
    report["ok"] = ok
    report["warnings"] = list(cfg.get("warnings", []))
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for name, r in report["devices"].items():
            print(f"[{name}] stream '{r['stream_name']}' | {r['n_channels']} canales a {r['fs']:g} Hz")
            for s in r["stages"]:
                state = f"ERROR: {s['error']}" if s["disabled"] else f"{s['avg_ms']:.3f} ms (máx {s['max_ms']:.3f})"
                print(f"    {s['stage']:<45} {state}")
            for e in r["errors"]:
                print(f"    ! {e}")
            print(f"    total {r['pipeline_avg_ms']:.3f} ms por ventana de {samples_per_window} muestras "
                  f"({r['window_ms']:.1f} ms de señal)")
        for w in report["warnings"]:
            print(f"AVISO: {w}")
        print("OK" if ok else "HAY ERRORES")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Puente UDP -> LSL v2 (varios dispositivos, pipeline por etapas).")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="ruta al bridge_config.json")
    ap.add_argument("--check", action="store_true",
                    help="valida la config, mide el pipeline con datos sintéticos y sale")
    ap.add_argument("--json", action="store_true", help="con --check: salida en JSON")
    ap.add_argument("--samples", type=int, default=20, help="con --check: muestras por ventana")
    ap.add_argument("--windows", type=int, default=200, help="con --check: ventanas de prueba")
    ap.add_argument("--debug", action="store_true", help="logs detallados")
    # Solo para el formato antiguo (un dispositivo); con "devices" se ignoran:
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--source-id", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="[%(levelname)s] %(name)s: %(message)s", stream=sys.stderr)

    overrides = {"host": args.host, "udp_port": args.port, "fs": args.fs, "n_channels": args.channels,
                 "stream_name": args.name, "source_id": args.source_id}
    if not any(v is not None for v in overrides.values()):
        overrides = None

    if args.check:
        try:
            cfg = load_config(args.config, overrides)
        except ConfigError as e:
            if args.json:
                print(json.dumps({"ok": False, "config_error": str(e)}, ensure_ascii=False))
            else:
                print(f"CONFIG INVÁLIDA: {e}")
            return 1
        return run_check(cfg, args.samples, args.windows, args.json)

    bridge = Bridge(args.config, overrides)
    try:
        bridge.start()
    except ConfigError as e:
        log.error("no se puede arrancar: %s", e)
        return 1

    def _on_signal(signum, _frame):
        log.info("señal %d recibida: cerrando", signum)
        bridge.stop_ev.set()

    os_signal.signal(os_signal.SIGTERM, _on_signal)
    os_signal.signal(os_signal.SIGINT, _on_signal)
    bridge.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
