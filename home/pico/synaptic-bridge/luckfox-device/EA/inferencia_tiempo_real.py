"""
Inferencia en tiempo real para la Luckfox usando solo NumPy (sin TensorFlow).

Archivos que deben estar en la misma carpeta que este script:
  - pesos.npz            (generado con exportar_pesos.py en la PC)
  - inferencia_numpy.py
  - EA_ref.npz
  - euclidean_align.py
  - carpeta synaptic/    (synaptic_bridge_lsl.py)
"""
import os
import time
from queue import Queue
from threading import Event, Thread

import numpy as np

from synaptic.inferencia_numpy import Modelo
from synaptic.euclidean_align import OnlineEuclideanAlignment
from synaptic.synaptic_bridge_lsl import LSLStream, send_output_to_socket


FS       = 500                          # muestras por segundo
WIN      = 100                          # tamaño de ventana = TIME_STEPS del modelo
OVERLAP  = 0.80
HOP      = int(WIN * (1 - OVERLAP))     # 20 muestras de avance
N_WIN    = 15                           # ventanas por batch
N_TOTAL  = WIN + (N_WIN - 1) * HOP      # 280 muestras necesarias
PERIODO  = N_WIN * HOP / FS             # 0.4 s de datos nuevos por ciclo

# Carpeta donde está este script (funciona igual en Windows y en la Luckfox)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

modelo = Modelo(os.path.join(MODEL_DIR, "modelos/GRUCNN.npz"))

stds_iniciales = np.array( [ 24.859, 493.139,  25.345,   7.263,  19.86,   22.22,   25.99,    6.408])

stds_finales = np.array([0.0231116 , 0.00385533, 0.03343345, 0.02997882, 0.04013269,0.04864398, 0.04540013, 0.03238339])

ea = OnlineEuclideanAlignment.load(os.path.join(MODEL_DIR, "datos/EA_ref.npz"),
                                   prior_count=30, forgetting="keep")


def inference_model(batch):
    return modelo.predecir(batch)       # (N_WIN, num_classes)


def make_windows(data, win, hop, n):
    """data: (muestras, canales) -> (n, win, canales)"""
    return np.stack([data[i * hop: i * hop + win] for i in range(n)]).astype(np.float32)


# --- Conectar LSL ---
emg = LSLStream(name="Synaptic Axon", buffer_secs=1)
if not emg.connect(timeout=5):
    raise RuntimeError("No se encontró el stream LSL.")
emg.wait_until_ready(timeout=5)

send_to_socket = True
if send_to_socket:
    stop_event = Event()
    output_queue = Queue()
    socket_thread = Thread(target=send_output_to_socket, args=(stop_event, output_queue))
    socket_thread.start()

print("Inferencia en tiempo real. Ctrl+C para detener.\n")

try:
    while True:
        inicio = time.time()

        times, data = emg.get_latest(N_TOTAL)        # (280, 8)
        if data.shape[0] < N_TOTAL:                  # aún no hay suficientes muestras
            time.sleep(0.01)
            continue

        data = np.asarray(data, dtype=np.float64)
        data = data - data.mean(axis=0)
        data = data / stds_iniciales
        data = np.abs(ea.partial_fit_transform(data))
        data = data / stds_finales   # (280, 8)
        batch = make_windows(data, WIN, HOP, N_WIN)  # (10, 100, 8)

        t0 = time.time()
        probs = inference_model(batch)
        t_inf = (time.time() - t0) * 1000

        idx  = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        print(idx, np.round(conf, 2), f"({t_inf:.0f} ms)")

        # if send_to_socket:
        #     valores, cuentas = np.unique(idx, return_counts=True)
        #     output_queue.put(int(valores[cuentas.argmax()]))   # voto mayoritario

        # Espera solo lo que falta para completar el periodo
        time.sleep(max(0.0, PERIODO - (time.time() - inicio)))

except KeyboardInterrupt:
    print("\nDetenido.")
    if send_to_socket:
        stop_event.set()
        socket_thread.join()
