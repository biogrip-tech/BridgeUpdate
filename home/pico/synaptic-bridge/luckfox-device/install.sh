#!/usr/bin/env bash
# install.sh — instala el puente UDP->LSL en ESTE Luckfox (Pico Ultra W, Ubuntu 22.04).
#
# Qué hace, en orden:
#   0. Prepara el sistema: amplía /run si hace falta y repara paquetes
#      que hayan quedado a medio instalar.
#   1. Instala dependencias de sistema (hostapd, dnsmasq, python3-pip,
#      numpy y scipy precompilados de Ubuntu).
#   2. Instala liblsl (librería nativa que necesita pylsl).
#   3. Instala las dependencias de Python restantes (pylsl, etc.).
#   4. Copia este directorio a /home/pico/synaptic-bridge/luckfox-device.
#   5. Crea /etc/synaptic-bridge/bridge_config.json a partir del ejemplo,
#      SI todavía no existe (nunca pisa una config ya personalizada).
#   6. Instala el arreglo del driver Wi-Fi (wifi-fix) para que wlan0
#      exista siempre al arrancar.
#   7. Instala y habilita el servicio systemd udp-lsl con sus ajustes.
#
# Lo que este script NO hace: generar la red única de cada dispositivo.
# Eso lo hace network/generate-device-network.sh (ver docs/04-configuracion-red.md).
#
# IMPORTANTE: ejecútalo con internet (wlan0 como cliente Wi-Fi o por otra vía).
# Si no hay internet, copia antes el .deb de liblsl a esta carpeta o a /tmp.
#
# Uso (en el Luckfox, como 'pico' con sudo o como root):
#   ./install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="/home/pico/synaptic-bridge/luckfox-device"
ETC_DIR="/etc/synaptic-bridge"
POLICY_FILE="/usr/sbin/policy-rc.d"

# liblsl precompilada para Ubuntu 22.04 (jammy) en ARM de 32 bits (armhf).
LIBLSL_VERSION="1.17.7"
LIBLSL_DEB="liblsl-${LIBLSL_VERSION}-jammy_arm.deb"
LIBLSL_URL="https://github.com/sccn/liblsl/releases/download/v${LIBLSL_VERSION}/${LIBLSL_DEB}"

# IP extra que se añade a wlan0 porque el firmware del ESP32 envía a esta
# dirección. Déjalo vacío ("") si el ESP32 ya envía a la IP propia del AP.
ESP32_TARGET_IP="192.168.4.1"

# Módulos del driver Wi-Fi AIC8800DC.
AIC_KO_DIR="/oem/usr/ko"

# Borra el bloqueo temporal de servicios aunque el script falle a medias.
cleanup() {
    if [ -f "${POLICY_FILE}" ] && grep -q "synaptic-bridge" "${POLICY_FILE}" 2>/dev/null; then
        sudo rm -f "${POLICY_FILE}"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
echo "[0/7] Preparando el sistema..."
# systemd necesita al menos 16 MB libres en /run para recargarse durante
# las actualizaciones (p. ej. libc6). En el Luckfox /run es muy pequeño.
RUN_FREE_KB=$(df -k --output=avail /run | tail -1 | tr -d ' ')
if [ "${RUN_FREE_KB}" -lt 32768 ]; then
    echo "[INFO] Poco espacio en /run (${RUN_FREE_KB} KB libres). Ampliando a 64M (temporal, hasta reiniciar)."
    sudo mount -o remount,size=64M /run
fi

# Repara paquetes que quedaron a medio configurar en intentos anteriores.
sudo dpkg --configure -a || true
sudo apt-get -f install -y || true

# ---------------------------------------------------------------------------
echo "[1/7] Instalando dependencias de sistema..."
# Evita que hostapd y dnsmasq arranquen al instalarse: sin configurar,
# dnsmasq choca con systemd-resolved en el puerto 53 y hace fallar apt.
printf '#!/bin/sh\n# synaptic-bridge: bloqueo temporal de servicios\nexit 101\n' | sudo tee "${POLICY_FILE}" > /dev/null
sudo chmod +x "${POLICY_FILE}"

sudo apt-get update
# numpy y scipy desde apt: vienen precompilados para armhf. Compilarlos
# con pip en este equipo tarda horas o se queda sin memoria.
sudo apt-get install -y python3 python3-pip python3-numpy python3-scipy \
    hostapd dnsmasq curl

cleanup
# Ubuntu instala hostapd enmascarado; lo desenmascaramos para que el
# servicio udp-lsl pueda arrancarlo.
sudo systemctl unmask hostapd || true

# ---------------------------------------------------------------------------
echo "[2/7] Instalando liblsl ${LIBLSL_VERSION}..."
if ldconfig -p | grep -q "liblsl"; then
    echo "[INFO] liblsl ya está instalada; no se reinstala."
else
    ARCH="$(dpkg --print-architecture)"
    if [ "${ARCH}" != "armhf" ]; then
        echo "[AVISO] Arquitectura ${ARCH}: este .deb es para armhf. Instala liblsl a mano."
    else
        # Busca primero un .deb copiado a mano (útil si no hay internet).
        DEB_PATH=""
        for candidate in "${SCRIPT_DIR}/${LIBLSL_DEB}" "/tmp/${LIBLSL_DEB}"; do
            if [ -f "${candidate}" ]; then
                DEB_PATH="${candidate}"
                break
            fi
        done
        if [ -z "${DEB_PATH}" ]; then
            DEB_PATH="/tmp/${LIBLSL_DEB}"
            curl -fL -o "${DEB_PATH}" "${LIBLSL_URL}"
        fi
        sudo apt-get install -y "${DEB_PATH}"
        sudo ldconfig
    fi
fi

# ---------------------------------------------------------------------------
echo "[3/7] Instalando dependencias de Python..."
# --break-system-packages solo existe en pip >= 23. Ubuntu 22.04 trae pip 22
# y no lo necesita, así que solo se usa si pip lo reconoce.
PIP_FLAGS=""
if python3 -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
    PIP_FLAGS="--break-system-packages"
fi

# numpy y scipy ya vienen de apt (paso 1). Si pip los ve en requirements.txt
# con una versión mayor, intenta compilarlos desde el código fuente, lo cual
# falla o tarda horas en este equipo. Se quitan de la lista para pip.
REQ_FILTERED="$(mktemp)"
grep -viE '^[[:space:]]*(numpy|scipy)([^a-zA-Z0-9_-]|$)' "${SCRIPT_DIR}/requirements.txt" > "${REQ_FILTERED}" || true
if [ -s "${REQ_FILTERED}" ]; then
    sudo python3 -m pip install ${PIP_FLAGS} -r "${REQ_FILTERED}"
fi
rm -f "${REQ_FILTERED}"
python3 -c "import numpy, scipy; print('[INFO] numpy', numpy.__version__, '/ scipy', scipy.__version__)"

if python3 -c "import pylsl; pylsl.library_version()" >/dev/null 2>&1; then
    echo "[INFO] pylsl y liblsl funcionan correctamente."
else
    echo "[AVISO] pylsl no encuentra liblsl. El servicio fallará hasta resolverlo."
fi

# ---------------------------------------------------------------------------
echo "[4/7] Copiando archivos a ${DEPLOY_DIR}..."
mkdir -p "${DEPLOY_DIR}"
if [ "$(realpath "${SCRIPT_DIR}")" != "$(realpath "${DEPLOY_DIR}")" ]; then
    cp -r "${SCRIPT_DIR}/." "${DEPLOY_DIR}/"
else
    echo "[INFO] Ya se está ejecutando desde ${DEPLOY_DIR}; no se copia nada."
fi
# Si se ejecuta como root, los archivos quedarían de root; se devuelven a pico.
sudo chown -R pico:pico /home/pico/synaptic-bridge

# ---------------------------------------------------------------------------
echo "[5/7] Preparando configuración en ${ETC_DIR}..."
sudo mkdir -p "${ETC_DIR}"
if [ ! -f "${ETC_DIR}/bridge_config.json" ]; then
    sudo cp "${SCRIPT_DIR}/config/bridge_config.example.json" "${ETC_DIR}/bridge_config.json"
    echo "[INFO] Creado ${ETC_DIR}/bridge_config.json (copiado del ejemplo)."
    echo "       Edítalo para poner un stream_name/source_id único de este usuario."
else
    echo "[INFO] ${ETC_DIR}/bridge_config.json ya existe; no se modifica."
fi
# El usuario 'pico' debe poder escribir aquí: el controlador de la PC edita
# este archivo por SFTP (sin sudo) y luego pide reiniciar el servicio.
sudo chown -R pico:pico "${ETC_DIR}"

# ---------------------------------------------------------------------------
echo "[6/7] Instalando arreglo del driver Wi-Fi (wifi-fix)..."
# El driver AIC8800DC suele fallar al arrancar por falta de memoria contigua
# ("Alloc bus->txdata_buf failed") y wlan0 no aparece. Este script espera a
# wlan0 y, si no aparece, recarga el driver tras liberar memoria.
sudo tee /usr/local/bin/wifi-fix.sh > /dev/null << EOF
#!/bin/sh
for i in \$(seq 1 30); do
    [ -e /sys/class/net/wlan0 ] && exit 0
    sleep 1
done
rmmod aic8800_fdrv 2>/dev/null
rmmod aic8800_btlpm 2>/dev/null
rmmod aic8800_bsp 2>/dev/null
sync
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory 2>/dev/null
insmod ${AIC_KO_DIR}/aic8800_bsp.ko
insmod ${AIC_KO_DIR}/aic8800_fdrv.ko
for i in \$(seq 1 15); do
    [ -e /sys/class/net/wlan0 ] && exit 0
    sleep 1
done
exit 1
EOF
sudo chmod +x /usr/local/bin/wifi-fix.sh

sudo tee /etc/systemd/system/wifi-fix.service > /dev/null << 'EOF'
[Unit]
Description=Carga el driver Wi-Fi AIC8800
Before=udp-lsl.service hostapd.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/wifi-fix.sh
TimeoutStartSec=90

[Install]
WantedBy=multi-user.target
EOF

# NetworkManager no debe tocar wlan0: lo controla hostapd como punto de acceso.
# No se recarga NetworkManager aquí para no cortar la conexión a internet
# durante la instalación; se aplica al reiniciar.
NM_CONF="/etc/NetworkManager/conf.d/99-wlan0-ap.conf"
if [ -d /etc/NetworkManager/conf.d ] && [ ! -f "${NM_CONF}" ]; then
    printf '[keyfile]\nunmanaged-devices=interface-name:wlan0\n' | sudo tee "${NM_CONF}" > /dev/null
    echo "[INFO] Creado ${NM_CONF}: NetworkManager dejará wlan0 al reiniciar."
fi

# ---------------------------------------------------------------------------
echo "[7/7] Instalando servicio systemd..."
sudo cp "${SCRIPT_DIR}/systemd/udp-lsl.service" /etc/systemd/system/udp-lsl.service

# Ajustes sin modificar el archivo del repo:
#  - espera a que wifi-fix deje wlan0 listo,
#  - reintenta sin límite si falla,
#  - añade la ruta multicast (sin ella LSL no encuentra streams, porque el
#    Luckfox en modo AP no tiene ruta por defecto),
#  - añade la IP a la que envía el ESP32 (si ESP32_TARGET_IP no está vacío).
sudo mkdir -p /etc/systemd/system/udp-lsl.service.d
{
    printf '[Unit]\nWants=wifi-fix.service\nAfter=wifi-fix.service\nStartLimitIntervalSec=0\n\n'
    printf '[Service]\nRestart=always\nRestartSec=10\n'
    printf 'ExecStartPre=-/sbin/ip route replace 224.0.0.0/4 dev ${AP_IFACE}\n'
    if [ -n "${ESP32_TARGET_IP}" ]; then
        printf 'ExecStartPre=-/sbin/ip addr replace %s/32 dev ${AP_IFACE}\n' "${ESP32_TARGET_IP}"
    fi
} | sudo tee /etc/systemd/system/udp-lsl.service.d/override.conf > /dev/null
sudo rm -f /etc/systemd/system/udp-lsl.service.d/multicast.conf

sudo systemctl daemon-reload
sudo systemctl enable wifi-fix.service udp-lsl.service

echo ""
echo "=== Instalación de software completa ==="
echo "Falta configurar la red única de este dispositivo (paso obligatorio):"
echo "  sudo ${DEPLOY_DIR}/network/generate-device-network.sh <nombre> <octeto_subred>"
echo ""
echo "Después reinicia para aplicar todo:"
echo "  sudo reboot"
echo ""
echo "Tras reiniciar, comprueba:"
echo "  systemctl status wifi-fix udp-lsl --no-pager"
echo "  sudo journalctl -u udp-lsl -f"