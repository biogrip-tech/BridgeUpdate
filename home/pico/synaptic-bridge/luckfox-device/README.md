# luckfox-device

Todo lo que corre **en el Luckfox**: el puente UDP→LSL, la cadena de
filtros, la config de ejemplo, el servicio systemd y las plantillas de red
por dispositivo.

```
luckfox-device/
├── udp_to_lsl_bridge.py         # entrypoint: UDP -> filtros -> LSL
├── filter_chain.py              # cadena de filtros IIR causales
├── requirements.txt             # pylsl, numpy, scipy
├── install.sh                   # instala todo lo de arriba como servicio
├── config/
│   └── bridge_config.example.json
├── systemd/
│   └── udp-lsl.service          # unidad genérica (misma en todos los dispositivos)
└── network/
    ├── hostapd.conf.example     # plantilla del AP WiFi
    ├── dnsmasq.conf.example     # plantilla del DHCP del AP
    ├── device.env.example       # plantilla de IP/subred que lee el .service
    └── generate-device-network.sh  # genera red ÚNICA para este dispositivo
```

Instalación completa y explicada: ver
[`docs/04-instalacion-servicio.md`](../docs/04-instalacion-servicio.md) y,
antes que nada, [`docs/05-configuracion-red.md`](../docs/05-configuracion-red.md)
(cada Luckfox necesita su propia red — no te lo saltes).

Quickstart:

```bash
./install.sh
sudo ./network/generate-device-network.sh <nombre_dispositivo> <octeto_subred>
sudo systemctl start udp-lsl
```
