#!/usr/bin/env python3
"""
lsl_diagnose.py — Diagnóstico de streams LSL desde la PC (o el Luckfox).

Responde a dos preguntas:
  1. ¿Hay varios streams LSL con el mismo nombre o source_id en la red?
     (un consumidor puede conectarse al equivocado o saltar a él al
     reconectarse, y parece que "la señal se cae").
  2. ¿Se pierden muestras y DÓNDE?
       - Filas con NaN: el puente rellenó paquetes que NO llegaron al Luckfox
         (Wi-Fi o ESP32). Requiere "timestamps": "counter" y "fill_gaps": true.
       - Huecos de timestamp sin NaN: con "timestamps": "arrival" las pérdidas
         del Wi-Fi se ven como saltos de tiempo.
       - Tasa < fs sin NaN ni huecos: el problema está entre el puente y el
         consumidor (red LSL, consumidor lento, stream recreado).
     Si existe el stream de estado del puente (SynapticBridge_status_*), se
     muestran también sus contadores por origen y por puerto.

Uso:
    python3 lsl_diagnose.py                     # lista y analiza todos los streams EMG 20 s
    python3 lsl_diagnose.py --seconds 60
    python3 lsl_diagnose.py --name "Synaptic Axon"
    python3 lsl_diagnose.py --list              # solo lista, no se conecta
"""

import argparse
import collections
import json
import time

import numpy as np
from pylsl import StreamInlet, resolve_streams


def describe(info):
    return (f"'{info.name()}' type={info.type()} source_id={info.source_id()} "
            f"host={info.hostname()} {info.channel_count()}ch@{info.nominal_srate():g}Hz "
            f"uid={info.uid()[:8]}")


def find_duplicates(infos):
    problems = []
    for key, label in ((lambda i: i.name(), "nombre"), (lambda i: i.source_id(), "source_id")):
        groups = collections.defaultdict(list)
        for i in infos:
            if key(i):
                groups[key(i)].append(i)
        for value, items in groups.items():
            if len(items) > 1:
                problems.append((label, value, items))
    return problems


def main():
    ap = argparse.ArgumentParser(description="Diagnóstico de streams LSL (duplicados y pérdidas).")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--name", default=None, help="analizar solo los streams con este nombre")
    ap.add_argument("--type", default="EMG", help="tipo de stream a analizar ('' = todos)")
    ap.add_argument("--wait", type=float, default=3.0, help="segundos buscando streams")
    ap.add_argument("--list", action="store_true", help="solo listar")
    args = ap.parse_args()

    infos = resolve_streams(wait_time=args.wait)
    print(f"== {len(infos)} streams LSL visibles")
    for i in sorted(infos, key=lambda x: (x.name(), x.hostname())):
        print("   ", describe(i))

    dups = find_duplicates(infos)
    print()
    if dups:
        print("== CONFLICTOS (un consumidor puede conectarse al stream equivocado):")
        for label, value, items in dups:
            hosts = ", ".join(sorted({i.hostname() for i in items}))
            print(f"   {len(items)} streams con el mismo {label} '{value}' (hosts: {hosts})")
    else:
        print("== Sin nombres ni source_id repetidos")
    if args.list:
        return 0

    targets = [i for i in infos
               if (not args.name or i.name() == args.name)
               and (not args.type or i.type() == args.type)]
    status = [i for i in infos if i.type() == "Status" and i.name().startswith("SynapticBridge_status")]
    if not targets:
        print("\nNo hay streams que analizar.")
        return 1

    inlets = []
    for info in targets:
        try:
            inl = StreamInlet(info, max_buflen=int(args.seconds) + 10, recover=True)
            inl.open_stream(timeout=5)
            inlets.append({"info": info, "inlet": inl, "n": 0, "nan": 0, "gaps": 0,
                           "gap_samples": 0, "last_ts": None, "max_gap": 0.0,
                           "backwards": 0})
        except Exception as e:
            print(f"   no se pudo abrir {describe(info)}: {e}")
    st_inlets = []
    for info in status:
        try:
            st_inlets.append((info, StreamInlet(info)))
        except Exception:
            pass

    print(f"\n== Midiendo {len(inlets)} streams durante {args.seconds:g} s ...")
    t_end = time.time() + args.seconds
    last_status = {}
    while time.time() < t_end:
        for s in inlets:
            data, ts = s["inlet"].pull_chunk(timeout=0.0, max_samples=4096)
            if not ts:
                continue
            a = np.asarray(data, dtype=np.float64)
            s["n"] += len(a)
            s["nan"] += int(np.isnan(a).any(axis=1).sum())
            fs = s["info"].nominal_srate() or 0
            t = np.asarray(ts)
            if s["last_ts"] is not None:
                t = np.concatenate(([s["last_ts"]], t))
            if fs > 0 and len(t) > 1:
                d = np.diff(t)
                s["backwards"] += int((d < 0).sum())
                big = d > 2.5 / fs
                s["gaps"] += int(big.sum())
                if big.any():
                    s["gap_samples"] += int(np.round(d[big] * fs - 1).sum())
                    s["max_gap"] = max(s["max_gap"], float(d.max()))
            s["last_ts"] = float(ts[-1])
        for info, inl in st_inlets:
            sample, _ = inl.pull_sample(timeout=0.0)
            if sample:
                last_status[info.hostname()] = sample[0]
        time.sleep(0.02)

    print()
    for s in inlets:
        info = s["info"]
        fs = info.nominal_srate()
        print(describe(info))
        rate = s["n"] / args.seconds
        print(f"    recibidas {s['n']} muestras ({rate:.1f}/s, esperadas ~{fs:g}/s)")
        print(f"    filas NaN (paquetes perdidos antes del Luckfox, rellenados): {s['nan']}")
        print(f"    huecos de timestamp > 2.5/fs: {s['gaps']} (~{s['gap_samples']} muestras, "
              f"mayor {s['max_gap'] * 1000:.0f} ms); timestamps hacia atrás: {s['backwards']}")
        if s["n"] == 0:
            print("    -> NO llega nada: ¿ESP32 apagado, stream sin consumidor válido o "
                  "conectado a un stream duplicado muerto?")
        elif fs and rate < 0.97 * fs and not s["nan"] and not s["gaps"]:
            print("    -> faltan muestras sin huecos marcados: revisa el arranque de la medida "
                  "o la conexión LSL (consumidor lento, stream recreado)")
        elif s["nan"] or s["gaps"]:
            print("    -> las pérdidas ocurren ANTES del puente (Wi-Fi / ESP32). Mira el log "
                  "del puente: 'paquetes NO llegaron al Luckfox'")
        else:
            print("    -> OK")

    if last_status:
        print("\n== Estado del puente (último informe)")
        for host, raw in last_status.items():
            try:
                st = json.loads(raw)
            except ValueError:
                continue
            print(f"  {host}:")
            for name, d in st.get("devices", {}).items():
                print(f"    [{name}] {d.get('rate_sps', 0):.0f} sps, perdidos={d.get('dropped')} "
                      f"cola_llena={d.get('queue_overflow')} resyncs={d.get('resyncs')} "
                      f"origen={d.get('sender')}")
            for port, p in st.get("ports", {}).items():
                print(f"    puerto {port}: descartes del kernel={p.get('kernel_drops')} "
                      f"ESP32 rechazados (catch-all ocupado)={p.get('foreign_packets')} "
                      f"sin dispositivo={p.get('unknown_packets')}")
            for c in st.get("lsl_conflicts", []):
                print(f"    CONFLICTO: {c}")
    elif not status:
        print("\n(no hay stream de estado del puente: pon \"status_stream\": true en la config "
              "para ver sus contadores aquí)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
