#!/usr/bin/env python3
"""
fake_esp32.py — Simula uno o varios ESP32 enviando EMG por UDP al puente.

Sirve para probar el puente (varios dispositivos, pérdidas, reinicios) sin
hardware. Usa el mismo formato que sendWindowUDP() del firmware.

Ejemplos:
    # Un ESP32 a 1000 Hz, 8 canales, 20 muestras por paquete:
    python3 fake_esp32.py --target 192.168.11.1

    # Simular un 2 % de paquetes perdidos:
    python3 fake_esp32.py --target 192.168.11.1 --loss 0.02

    # Dos "ESP32" desde la misma máquina (Linux): cada uno con su IP de origen
    # en loopback, y el puente en esta misma máquina con match por IP:
    python3 fake_esp32.py --target 127.0.0.1 --bind-ip 127.0.0.2 &
    python3 fake_esp32.py --target 127.0.0.1 --bind-ip 127.0.0.3 --freq 30
"""

import argparse
import random
import socket
import struct
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Simulador de ESP32 EMG por UDP.")
    ap.add_argument("--target", default="127.0.0.1", help="IP del puente")
    ap.add_argument("--port", type=int, default=50043)
    ap.add_argument("--bind-ip", default=None, help="IP de origen (para simular varios ESP32)")
    ap.add_argument("--fs", type=float, default=300.0)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--samples", type=int, default=20, help="muestras por paquete")
    ap.add_argument("--freq", type=float, default=60.0, help="frecuencia de la senoide de prueba (Hz)")
    ap.add_argument("--loss", type=float, default=0.0, help="fracción de paquetes a 'perder'")
    ap.add_argument("--seconds", type=float, default=0.0, help="duración (0 = infinito)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.bind_ip:
        sock.bind((args.bind_ip, 0))
    period = args.samples / args.fs
    counter, n = 0, 0
    t0 = time.monotonic()
    rng = np.random.default_rng()
    print(f"enviando a {args.target}:{args.port} | {args.channels} canales, {args.fs:g} Hz, "
          f"{args.samples} muestras/paquete ({1 / period:.1f} paquetes/s)")
    try:
        while True:
            t = (n + np.arange(args.samples)) / args.fs
            sine = 100.0 * np.sin(2 * np.pi * args.freq * t)[:, None]
            offset = np.arange(args.channels)[None, :] * 10.0     # DC distinto por canal
            noise = rng.standard_normal((args.samples, args.channels)) * 20.0
            window = (sine + offset + noise).astype("<f4")
            if random.random() >= args.loss:
                header = struct.pack("<IHH", counter & 0xFFFFFFFF, args.channels, args.samples)
                sock.sendto(header + window.tobytes(), (args.target, args.port))
            counter += 1
            n += args.samples
            if args.seconds and time.monotonic() - t0 >= args.seconds:
                break
            sleep = t0 + counter * period - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    print(f"enviados {counter} paquetes")


if __name__ == "__main__":
    main()
