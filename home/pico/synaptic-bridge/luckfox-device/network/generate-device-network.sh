#!/usr/bin/env bash
# generate-device-network.sh — genera la configuración de red ÚNICA de este
# Luckfox a partir de las plantillas *.example de este mismo directorio.
#
# Por qué existe: cada Luckfox que se flashea debe salir con un SSID, una
# subred y un canal WiFi distintos a los de cualquier otro Luckfox ya
# desplegado, para que:
#   - Varias unidades puedan encenderse en el mismo espacio sin interferirse.
#   - La PC de control nunca ambigüe a qué Luckfox se está conectando.
#   - Los streams LSL de distintos usuarios no choquen en la misma red.
# Este script deja esa decisión en UN solo número por dispositivo (el octeto
# de subred) en vez de tener que tocar 3 archivos de config a mano.
#
# Uso (en el Luckfox, como root o con sudo):
#   sudo ./generate-device-network.sh <nombre_dispositivo> <octeto_subred> [canal]
#
# Ejemplo:
#   sudo ./generate-device-network.sh synaptic-fer 7 6
#   -> SSID:   Synaptic-synaptic-fer
#   -> Subred: 192.168.7.0/24 (AP en 192.168.7.1)
#   -> Canal:  6
#
# El octeto de subred (2do argumento) es el identificador corto del
# dispositivo: anótalo en un inventario (ver docs/04-configuracion-red.md)
# para no repetirlo nunca entre Luckfox distintos.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Uso: $0 <nombre_dispositivo> <octeto_subred 1-254> [canal 1-11] [passphrase]" >&2
    exit 1
fi

DEVICE_NAME="$1"
SUBNET_OCTET="$2"
AP_CHANNEL="${3:-6}"
AP_PASSPHRASE="${4:-}"

if ! [[ "$SUBNET_OCTET" =~ ^[0-9]+$ ]] || [ "$SUBNET_OCTET" -lt 1 ] || [ "$SUBNET_OCTET" -gt 254 ]; then
    echo "ERROR: el octeto de subred debe ser un número entre 1 y 254." >&2
    exit 1
fi

if [ -z "$AP_PASSPHRASE" ]; then
    # Genera una passphrase aleatoria de 12 caracteres si no se dio ninguna.
    AP_PASSPHRASE="$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 12)"
    echo "[INFO] No se dio passphrase; generada aleatoriamente: $AP_PASSPHRASE"
fi

AP_SSID="Synaptic-${DEVICE_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

render() {
    local template="$1" dest="$2"
    sed \
        -e "s/__AP_IFACE__/wlan0/g" \
        -e "s/__AP_SSID__/${AP_SSID}/g" \
        -e "s/__AP_CHANNEL__/${AP_CHANNEL}/g" \
        -e "s/__AP_PASSPHRASE__/${AP_PASSPHRASE}/g" \
        -e "s/__SUBNET_OCTET__/${SUBNET_OCTET}/g" \
        -e "s/__DEVICE_NAME__/${DEVICE_NAME}/g" \
        "$template" > "$dest"
    echo "[OK] Escrito $dest"
}

sudo mkdir -p /etc/synaptic-bridge /etc/hostapd

render "${SCRIPT_DIR}/hostapd.conf.example"  /tmp/hostapd.conf.generated
render "${SCRIPT_DIR}/dnsmasq.conf.example"  /tmp/dnsmasq.conf.generated
render "${SCRIPT_DIR}/device.env.example"    /tmp/device.env.generated

sudo mv /tmp/hostapd.conf.generated /etc/hostapd/hostapd.conf
sudo mv /tmp/dnsmasq.conf.generated /etc/dnsmasq.conf
sudo mv /tmp/device.env.generated   /etc/synaptic-bridge/device.env

echo ""
echo "=== Configuración de red generada para '${DEVICE_NAME}' ==="
echo "SSID:        ${AP_SSID}"
echo "Passphrase:  ${AP_PASSPHRASE}"
echo "Subred:      192.168.${SUBNET_OCTET}.0/24 (AP en 192.168.${SUBNET_OCTET}.1)"
echo "Canal:       ${AP_CHANNEL}"
echo ""
echo "Guarda estos datos: la PC de control los necesita para conectarse."
echo "Siguiente paso: sudo systemctl restart hostapd dnsmasq udp-lsl"
