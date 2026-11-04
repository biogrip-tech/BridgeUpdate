"""
envolvente_rms.py — Plugin de ejemplo: envolvente RMS móvil (causal).

Muestra el contrato de los plugins del puente v2:
  - init(fs, n_channels, params) -> estado   (se llama una vez)
  - process(window, state)      -> ventana   (se llama con cada paquete)

`window` es (n_muestras, n_canales) en float64 y solo contiene los canales
seleccionados en la etapa ("channels" en la config). Hay que devolver un
array de la MISMA forma. El estado se guarda entre llamadas, así que la
ventana móvil continúa sin cortes de un paquete al siguiente.

Uso en la config:
    {"type": "custom", "file": "envolvente_rms.py", "params": {"window_ms": 100}}

Pruébalo antes de desplegar:
    python3 udp_to_lsl_bridge.py --config bridge_config.json --check
"""

import numpy as np
from scipy import signal


def init(fs, n_channels, params):
    n = max(2, int(round(float(params.get("window_ms", 100)) * fs / 1000.0)))
    return {
        "b": np.ones(n) / n,                 # media móvil de n muestras
        "zi": np.zeros((n - 1, n_channels)),  # estado del filtro, por canal
    }


def process(window, state):
    squared = window * window
    mean_sq, state["zi"] = signal.lfilter(state["b"], [1.0], squared, axis=0, zi=state["zi"])
    return np.sqrt(np.maximum(mean_sq, 0.0))
