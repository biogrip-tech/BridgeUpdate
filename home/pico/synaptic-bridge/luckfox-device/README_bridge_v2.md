# Puente UDP → LSL v2

El puente recibe las ventanas EMG que envían los ESP32 por UDP, las procesa con
un pipeline configurable y las publica en LSL. Esta versión admite varios
dispositivos a la vez, publica un stream crudo y otro procesado por
dispositivo, y permite filtros por canal, CAR, blanqueo espacial y funciones
de Python propias.

## Archivos

| Archivo | Qué hace |
| --- | --- |
| `udp_to_lsl_bridge.py` | Programa principal (receptores, dispositivos, LSL, recarga en caliente) |
| `pipeline.py` | Etapas del pipeline y su validación |
| `filter_chain.py` | Diseño de los filtros IIR (lo usa `pipeline.py`) |
| `euclidean_align.py` | Blanqueo / alineamiento euclidiano (lo usa la etapa `whitening`) |
| `envolvente_rms.py` | Plugin de ejemplo (se instala en `/etc/synaptic-bridge/plugins/`) |
| `config/bridge_config.example.json` | Default v2 de un dispositivo (lo copia `install.sh`) |
| `bridge_config.example.json` | Ejemplo multi-dispositivo |
| `fake_esp32.py` | Simulador de ESP32 para hacer pruebas sin hardware |
| `lsl_diagnose.py` | Diagnóstico desde la PC: streams duplicados y pérdidas por stream |

Los cuatro primeros deben estar juntos en la misma carpeta del Luckfox.

## Cómo funciona

```
ESP32 ──UDP──► [hilo de recepción del puerto] ──cola──► [hilo del dispositivo] ──► LSL <nombre>_raw
ESP32 ──UDP──►        (reparte por IP)        ──cola──► [hilo del dispositivo] ──► LSL <nombre>
```

Cada puerto UDP tiene un hilo que solo lee el socket y reparte los paquetes por
IP de origen. Cada dispositivo tiene su propia cola y su propio hilo, que
detecta pérdidas, calcula timestamps, ejecuta el pipeline y publica. Si el
pipeline de un dispositivo va lento, el socket se sigue vaciando y los demás
dispositivos no se ven afectados.

## Formato de la config

```json
{
  "bridge": { "...ajustes globales..." },
  "devices": [ { "...un objeto por ESP32..." } ]
}
```

Las claves que empiezan por `_` se ignoran, así que se pueden usar como
comentarios (`"_nota": "..."`). Si la config no tiene `devices`, se interpreta
en el formato antiguo (el que escribe la GUI actual): un único dispositivo
con los filtros fijos, sin stream crudo.

### `bridge` (todo opcional)

| Clave | Por defecto | Significado |
| --- | --- | --- |
| `host` | `"0.0.0.0"` | IP local donde escuchar |
| `rcvbuf_bytes` | `4194304` | Búfer del socket UDP; requiere subir `net.core.rmem_max` |
| `queue_size` | `2000` | Paquetes en espera por dispositivo antes de descartar |
| `max_buffered_s` | `30` | Segundos que LSL guarda por consumidor (memoria del Luckfox) |
| `stats_interval_s` | `5` | Cada cuánto se registran estadísticas |
| `stage_time_budget_ms` | `5` | Aviso si una etapa tarda más de esto por paquete |
| `status_stream` | `true` | Publicar el stream de estado |
| `status_stream_name` | `SynapticBridge_status_<host>` | Nombre del stream de estado |
| `discovery_enabled` | `true` | Escribir el JSON de descubrimiento de orígenes UDP |
| `discovery_path` | `/tmp/synaptic-bridge-discovery.json` | Ruta del JSON de descubrimiento |
| `discovery_interval_s` | `1` | Cada cuánto se reescribe el JSON |
| `discovery_stale_s` | `120` | Segundos sin paquetes para olvidar un origen |
| `catch_all_hold_s` | `3` | Un dispositivo con `match` vacío se engancha al primer ESP32 que le envía; otra IP en ese puerto se descarta hasta que el primero lleve este tiempo sin enviar |
| `lsl_conflict_check_s` | `30` | Cada cuánto se buscan en la red streams LSL con el mismo nombre o `source_id` que los nuestros (`0` = no buscar) |
| `plugins_dir` | `<carpeta de la config>/plugins` | Dónde buscar los plugins |
| `data_dir` | `<carpeta de la config>` | Dónde buscar las referencias `.npz` |

### `devices`

| Clave | Por defecto | Significado |
| --- | --- | --- |
| `name` | obligatorio | Nombre único del dispositivo (logs y estado) |
| `match` | `{}` | `{"ip": "192.168.11.27"}`; vacío = cualquier IP en ese puerto |
| `udp_port` | `50043` | Puerto al que envía el ESP32 |
| `fs` | `1000` | Frecuencia de muestreo real del ESP32 (Hz) |
| `n_channels` | `8` | Canales esperados (se ajusta al valor real del firmware) |
| `stream_name` | `BioMyo_EMG_<name>` | Stream procesado; el crudo es `<stream_name>_raw` |
| `stream_type` | `"EMG"` | Tipo LSL |
| `source_id` | `<host>-<name>` | Identificador único; el crudo usa `<source_id>_raw` |
| `channel_labels` | `EMG_1..EMG_N` | Lista de nombres de canal |
| `publish_raw` | `true` | Publicar también el stream crudo |
| `timestamps` | `"counter"` | `"counter"` (sin jitter) o `"arrival"` (hora de llegada) |
| `fill_gaps` | `true` | Rellenar con `NaN` las muestras de paquetes perdidos |
| `max_gap_packets` | `100` | Un salto mayor se trata como reinicio del ESP32 |
| `pipeline` | `[]` | Lista de etapas (ver abajo) |

Reglas que se validan al cargar: nombres y `source_id` únicos, y como mucho
un dispositivo por cada combinación de puerto e IP (incluido un solo
dispositivo "cualquier IP" por puerto).

Para que `match.ip` sea estable, fija la IP de cada ESP32 con una reserva
DHCP en la config de dnsmasq del Luckfox, usando su MAC (aparece en
`/var/lib/misc/dnsmasq.leases`):

```
dhcp-host=34:85:18:03:96:9c,192.168.11.27
```

## Etapas del pipeline

Todas admiten `"channels"` (`"all"` o lista de canales **empezando en 1**,
como las etiquetas `EMG_1..EMG_N`), `"enabled": false` y `"label"`. Los canales
no seleccionados pasan sin cambios por esa etapa.

| Tipo | Parámetros |
| --- | --- |
| `highpass`, `lowpass` | `cutoff` (Hz), `family` (`butter`, `cheby1`, `cheby2`, `ellip`), `order` (1–12), `ripple`, `attenuation` |
| `bandpass`, `bandstop` | `low`, `high` (Hz), `family`, `order`, `ripple`, `attenuation` |
| `notch` | `freq` (Hz), `quality`, `harmonics` (cuántos múltiplos de `freq` eliminar) |
| `car` | Resta a cada canal la media de los canales seleccionados |
| `whitening` | `time_constant` (s), `refresh_every`, `shrinkage`, `subtract_mean`, `reference` (`.npz`), `adapt` |
| `custom` / `plugin` | `file` (`.py` en `plugins_dir`), `params` (se pasan a `init`) |

Notas:

- Los filtros IIR son causales y guardan su estado entre paquetes, como en v1.
  Ahora el estado inicial se ajusta al primer valor de cada canal, así que
  no hay un transitorio grande al arrancar.
- Pon `car` **después** del pasa-altas: un offset de DC en un electrodo se
  arrastraría a todos los canales.
- `whitening` usa `StreamingWhitener` de `euclidean_align.py`. Con
  `reference` arranca desde un `.npz` guardado con
  `EuclideanAlignment.save()`; con `"adapt": false` usa esa referencia fija.
  Esta etapa **mezcla canales**: tras ella, el canal 3 ya no es el electrodo 3.
- Las frecuencias se validan contra `fs/2`. Una etapa mal configurada no se
  crea, se registra el error y el resto del pipeline funciona.

## Plugins

Un plugin es un archivo `.py` en `plugins_dir`:

```python
def init(fs, n_channels, params):   # opcional, se llama una vez
    return {}                        # estado: cualquier objeto

def process(window, state):         # obligatorio, se llama con cada paquete
    return window                    # misma forma que la entrada
```

`window` tiene forma `(n_muestras, n_canales_seleccionados)`, en float64, y es
una copia. Ver el plugin de ejemplo `envolvente_rms.py`.

Si un plugin lanza una excepción, devuelve una forma distinta o produce
`NaN`/`Inf`, esa etapa se desactiva y la señal sigue sin ella. El stream crudo
no se ve afectado nunca. El Luckfox tiene numpy 1.21 y scipy 1.8, así que un
plugin que use funciones más nuevas fallará allí aunque funcione en la PC.

## Recarga en caliente

El puente revisa la config cada segundo, y también los plugins y las
referencias `.npz` que use. Lo que pasa según lo que cambie:

- **Solo el `pipeline` de un dispositivo, o un plugin:** se reconstruye ese
  pipeline. Se recrea solo el stream procesado (con el mismo `source_id`, así
  que los consumidores se reconectan solos). El stream crudo no se toca, de
  modo que una grabación de LabRecorder no se corta.
- **Cualquier otra clave del dispositivo:** se recrean sus dos streams.
- **Dispositivos añadidos o eliminados, o puertos nuevos:** se crean o cierran.
- **Config inválida:** se registra el error y se mantiene la anterior.

Ya no hace falta `systemctl restart udp-lsl` después de editar la config.

## Validar antes de desplegar

```bash
python3 udp_to_lsl_bridge.py --config /etc/synaptic-bridge/bridge_config.json --check
python3 udp_to_lsl_bridge.py --config /etc/synaptic-bridge/bridge_config.json --check --json
```

Construye cada pipeline, lo ejecuta con datos sintéticos y muestra el tiempo
de cada etapa y los errores, sin abrir sockets ni streams. Termina con código
0 si todo está bien y 1 si hay errores, así que la GUI puede usarlo por SSH
antes de subir una config.

## Stream de estado

Cada `stats_interval_s`, el puente publica en `SynapticBridge_status_<host>` un
JSON con, por dispositivo: tasa real (`rate_sps`), paquetes, perdidos y
porcentaje, desbordamientos de cola, paquetes inválidos, IP de origen,
segundos desde el último paquete, y tiempo medio y máximo de cada etapa.
Es lo que usará la GUI para mostrar el estado en vivo.

## Descubrimiento de orígenes UDP

El puente registra cada combinación IP:puerto que le envía paquetes, aunque no
esté en `devices`. No publica streams provisionales ni modifica la config:
solo escribe un JSON (por defecto `/tmp/synaptic-bridge-discovery.json`) cada
`discovery_interval_s`. Cada origen incluye IP, puerto, canales, muestras por
paquete, tasa estimada, antigüedad y el dispositivo de la config al que quedó
asignado.

Esto permite que la GUI detecte un segundo equipo en un puerto que ya tiene un
dispositivo catch-all (`"match": {}`), sin exponer IPs ni puertos al usuario.
Si un origen deja de enviar durante `discovery_stale_s`, desaparece del JSON.

## Diagnóstico de pérdidas con varios ESP32

**Un dispositivo con `"match": {}` acepta UN solo ESP32.** Antes, si dos ESP32
enviaban al mismo puerto y la config tenía un único dispositivo catch-all,
sus paquetes se mezclaban en el mismo stream: el contador saltaba hacia
delante y atrás en cada paquete (re-sincronizaciones continuas, ~90 % de
"perdidos" en las estadísticas, timestamps rotos y el estado de los filtros
contaminado). Ahora el catch-all se engancha al primero y el resto se
descarta con un aviso en el log. Con varios ESP32, crea un dispositivo por
ESP32 con su `match.ip` (y una reserva DHCP para cada uno), o usa un puerto
distinto por ESP32.

**Una sola instancia del puente por Luckfox.** El socket ya no usa
`SO_REUSEADDR`: antes, una segunda instancia (el servicio más una ejecución a
mano, o un proceso colgado) abría el mismo puerto sin error y se quedaba con
los paquetes, mientras la primera seguía publicando un stream LSL con el
mismo nombre y `source_id`, vivo pero vacío. Ahora la segunda instancia no
arranca y lo dice (`sudo ss -ulpn | grep 50043` muestra quién tiene el puerto).

**Dónde se pierde cada muestra.** El log (y el stream de estado) separa las
causas:

| Mensaje / contador | Dónde se perdió |
| --- | --- |
| `origen IP:puerto: N de M paquetes NO llegaron al Luckfox` | Wi-Fi o ESP32 (el contador del paquete tiene huecos al recibirlo) |
| `el kernel descartó N datagramas` (`ports.kernel_drops`) | Búfer del socket lleno: el puente no lee a tiempo |
| `cola_llena` (`queue_overflow`) | El pipeline del dispositivo es más lento que los datos |
| `ESP32 rechazados` (`ports.foreign_packets`) | Un segundo ESP32 en un puerto catch-all ya ocupado |
| `CONFLICTO LSL` | Otro stream en la red con el mismo nombre o `source_id` |

Si nada de eso aparece y el consumidor aun así pierde muestras, el problema
está en el lado LSL/PC. Para comprobarlo desde la PC:

```bash
python3 lsl_diagnose.py --seconds 30
```

Lista todos los streams, marca nombres y `source_id` repetidos, y mide en cada
stream la tasa real, las filas `NaN` (paquetes rellenados por el puente) y
los huecos de timestamp. Para que los huecos se rellenen con `NaN` hace falta
`"timestamps": "counter"` y `"fill_gaps": true`; con `"arrival"` las pérdidas
solo se ven como saltos de tiempo.

Usa `stream_name` y `source_id` únicos por Luckfox **y** por dispositivo. Si
todos los Luckfox tienen el mismo `source_id` (por ejemplo
`"synaptic-bridge"`) o el mismo hostname, un consumidor que se reconecta puede
acabar en el stream de otro Luckfox.

## Pruebas sin hardware

En la PC o en el Luckfox:

```bash
python3 fake_esp32.py --target 192.168.11.1                 # un ESP32
python3 fake_esp32.py --target 192.168.11.1 --loss 0.02     # con 2 % de pérdidas
```

Para simular dos ESP32 desde una sola máquina Linux (IPs de origen distintas):

```bash
python3 fake_esp32.py --target 127.0.0.1 --bind-ip 127.0.0.2 &
python3 fake_esp32.py --target 127.0.0.1 --bind-ip 127.0.0.3 &
```

con dos dispositivos en la config con `match.ip` `127.0.0.2` y `127.0.0.3`.

## Migrar un Luckfox existente

1. Copia `udp_to_lsl_bridge.py`, `pipeline.py`, `filter_chain.py` y
   `euclidean_align.py` a `/home/pico/synaptic-bridge/luckfox-device/`.
2. Crea `/etc/synaptic-bridge/plugins/` y copia ahí los plugins.
3. Escribe la config nueva en `/etc/synaptic-bridge/bridge_config.json` y
   valídala con `--check`.
4. `sudo systemctl restart udp-lsl` (solo esta vez).

GUI11 ya escribe y valida el formato v2. En GUIs anteriores, **no uses su
botón "Aplicar"** en un Luckfox con config v2: escribirían el formato antiguo
y sobrescribirían la config de varios dispositivos.
