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
    for key in ("max_buffered_s", "stats_interval_s", "stage_time_budget_ms"):
        g[key] = float(g[key])

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

    return {"bridge": g, "devices": devices, "legacy": legacy}


def device_identity(dcfg, gcfg):
    """Todo lo que, si cambia, obliga a recrear el dispositivo (no el pipeline)."""
    ident = {k: v for k, v in dcfg.items() if k != "pipeline"}
    ident["_global"] = {k: gcfg[k] for k in WORKER_GLOBAL_KEYS}
    return json.dumps(ident, sort_keys=True)


# ═══════════════════════════════════════════════════════════════════════════
# Utilidades
# ═══════════════════════════════════════════════════════════════════════════
def parse_packet(packet):
    """Devuelve (counter, ventana float32 (n_muestras, n_canales)) o None."""
    if len(packet) < HEADER_SIZE:
        return None
    counter, n_ch, n_samp = struct.unpack_from(HEADER_FMT, packet, 0)
    if n_ch <= 0 or n_ch > 64 or n_samp <= 0 or n_samp > 100000:
        return None
    avail_floats = (len(packet) - HEADER_SIZE) // 4
    use = min(n_samp * n_ch, avail_floats)
    use -= use % n_ch
    if use <= 0:
        return None
    floats = np.frombuffer(packet, dtype=np.float32, count=use, offset=HEADER_SIZE)
    return counter, floats.reshape(use // n_ch, n_ch)


def push_chunk(outlet, data, ts):
    """Publica un bloque (n_muestras, n_canales). Usa numpy directamente si la
    versión de pylsl lo admite (mucho más rápido que .tolist())."""
    arr = np.ascontiguousarray(data, dtype=np.float32)
    try:
        outlet.push_chunk(arr, ts)
    except (TypeError, ValueError):
        outlet.push_chunk(arr.tolist(), ts)


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
                self._handle(*item)
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

    def _handle(self, t_arrival, sender_ip, packet):
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
# Recepción: un hilo por puerto UDP, solo lee y reparte
# ═══════════════════════════════════════════════════════════════════════════
class Receiver(threading.Thread):
    UNKNOWN_LOG_EVERY_S = 30.0

    def __init__(self, host, port, rcvbuf):
        super().__init__(daemon=True, name=f"rx-{port}")
        self.host, self.port = host, port
        self._stop_ev = threading.Event()
        self.routes = ({}, None)       # (ip -> worker, worker para cualquier IP)
        self._unknown_last = {}
        self.unknown_packets = 0

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(rcvbuf))
        except OSError:
            pass
        sock.bind((host, port))
        sock.settimeout(0.5)
        self.sock = sock
        actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        # Linux reporta el doble del valor efectivo pedido; si se queda corto
        # es porque net.core.rmem_max lo limita.
        if actual < rcvbuf:
            log.warning("puerto %d: búfer de recepción de %d KB (se pidieron %d KB). "
                        "Sube net.core.rmem_max (ver README) para evitar pérdidas.",
                        port, actual // 1024, rcvbuf // 1024)
        log.info("escuchando UDP en %s:%d", host, port)

    def set_routes(self, by_ip, catch_all):
        self.routes = (dict(by_ip), catch_all)    # asignación atómica

    def stop(self):
        self._stop_ev.set()

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
            by_ip, catch_all = self.routes
            worker = by_ip.get(addr[0]) or catch_all
            if worker is None:
                self._log_unknown(addr[0])
                continue
            worker.submit((t_arrival, addr[0], data))
        try:
            self.sock.close()
        except OSError:
            pass

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
        self._cfg_mtime = None
        self._watched = {}       # nombre -> {archivo: mtime}

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
        for port in list(self.receivers):
            if port not in ports or rebuild_all:
                self.receivers.pop(port).stop()
        for port in sorted(ports):
            if port in self.receivers:
                continue
            try:
                r = Receiver(g["host"], port, g["rcvbuf_bytes"])
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
        while not self.stop_ev.wait(1.0):
            self._check_reload()
            now = time.monotonic()
            if now >= next_stats:
                self._report(now - last_stats)
                last_stats = now
                interval = self.cfg["bridge"]["stats_interval_s"]
                next_stats = now + interval
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
