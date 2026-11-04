"""
Calibración + fine-tuning en la Luckfox (solo NumPy).

1. Graba cada clase varias veces desde el stream LSL.
2. Calcula la desviación estándar por canal de los datos nuevos.
3. Calcula una referencia nueva de alineamiento euclidiano (EA) con los datos
   nuevos ya normalizados, y la guarda en EA_calibracion.npz.
4. Preprocesa igual que en tiempo real (media, stds nuevas, EA nueva, abs, ventanas).
5. Extrae la penúltima capa del CNN-GRU (128 características).
6. Entrena una MLP 128 -> oculta -> num_clases y la guarda en finetune.npz.

Uso:
  python3 calibrar.py                 # grabar y entrenar
  python3 calibrar.py --solo-entrenar # reentrenar con la última grabación guardada
"""
import os
import sys
import time
import numpy as np

from synaptic.inferencia_numpy import Modelo
from synaptic.mlp_numpy import MLP
from synaptic.euclidean_align import EuclideanAlignment, OnlineEuclideanAlignment

# ---------------- Configuración ----------------
FS       = 500
WIN      = 100
HOP      = 20
N_WIN    = 10
N_TOTAL  = WIN + (N_WIN - 1) * HOP      # 280
PASO     = N_WIN * HOP                  # 200 muestras nuevas por ciclo (igual que en tiempo real)

CLASES        = None    # p. ej. ["g0", "g1", "g2", ...]; None = una por salida del modelo, en orden
REPETICIONES  = 1       # veces que se graba cada clase (la última se usa para validar)
SEG_GESTO     = 7.0     # segundos de grabación por repetición
SEG_PREPARAR  = 3.0     # cuenta regresiva antes de cada grabación
DESCARTAR_S   = 1.0     # se descarta el inicio de cada grabación (tiempo de reacción)
N_OCULTA      = 120

DIR          = os.path.dirname(os.path.abspath(__file__))
RUTA_PESOS   = os.path.join(DIR, "modelos/GRUCNN.npz")
RUTA_EA_NUEVA = os.path.join(DIR, "datos/EA_calibracion.npz")   # EA_ref.npz original no se toca
RUTA_DATOS   = os.path.join(DIR, "datos/calibracion_datos.npz")
RUTA_SALIDA  = os.path.join(DIR, "datos/finetune.npz")
# ------------------------------------------------

gestos = {"g0":"Descanso", "g1": "Pulgar arriba", "g2": "Dos dedos", "g3": "Punio cerrado", "g4":"un dedo", "g5":"Ext Munieca", "g6":"Flex Munieca", "g7":"Abrir Mano", "g8":"pinch"}


def make_windows(data, win=WIN, hop=HOP, n=N_WIN):
    return np.stack([data[i * hop: i * hop + win] for i in range(n)]).astype(np.float32)


def calcular_stds(ensayos):
    """Desviación estándar por canal de todos los ensayos (quitando la media de cada uno)."""
    centrados = [e - e.mean(axis=0) for e in ensayos]
    return np.concatenate(centrados).std(axis=0)


def bloques_normalizados(raw, stds):
    """Bloques de 280 muestras cada 200, centrados y divididos por las stds (como en tiempo real)."""
    for ini in range(0, len(raw) - N_TOTAL + 1, PASO):
        bloque = raw[ini:ini + N_TOTAL].astype(np.float64)
        bloque = bloque - bloque.mean(axis=0)
        yield bloque / stds


def ventanas_de_ensayo(raw, stds, ea):
    """Reproduce el preprocesamiento de tiempo real sobre una grabación completa."""
    salida = []
    for bloque in bloques_normalizados(raw, stds):
        bloque = np.abs(ea.partial_fit_transform(bloque))
        salida.append(make_windows(bloque))
    if not salida:
        return np.zeros((0, WIN, raw.shape[1]), np.float32)
    return np.concatenate(salida)


def grabar(emg, segundos):
    """Acumula las muestras nuevas del stream durante 'segundos'."""
    t, _ = emg.get_latest(1)
    ultimo = np.asarray(t).ravel()[-1] if len(t) else -np.inf
    partes = []
    fin = time.time() + segundos
    while time.time() < fin:
        t, d = emg.get_latest(FS)
        t, d = np.asarray(t).ravel(), np.asarray(d)
        if len(t):
            nuevos = t > ultimo
            if nuevos.any():
                partes.append(d[nuevos])
                ultimo = t[nuevos][-1]
        time.sleep(0.05)
    return np.concatenate(partes) if partes else np.zeros((0, 8))


def sesion_de_grabacion(clases):
    from synaptic.synaptic_bridge_lsl import LSLStream

    emg = LSLStream(name="Synaptic Axon", buffer_secs=2)
    if not emg.connect(timeout=5):
        raise RuntimeError("No se encontró el stream LSL.")
    emg.wait_until_ready(timeout=5)

    datos, etiquetas, ensayo_id, repeticion = [], [], [], []
    n_ensayo = 0
    for rep in range(REPETICIONES):
        for c, nombre in enumerate(clases):
            print(f"\n[{rep + 1}/{REPETICIONES}] Prepárate para: {gestos[nombre]}")
            for s in range(int(SEG_PREPARAR), 0, -1):
                print(f"  {s}...")
                time.sleep(1)
            print(f"  ¡AHORA! Mantén '{gestos[nombre]}' {SEG_GESTO:.0f} s")
            raw = grabar(emg, SEG_GESTO)
            print(f"  Listo ({len(raw)} muestras). Descansa.")
            datos.append(raw)
            etiquetas.append(np.full(len(raw), c))
            ensayo_id.append(np.full(len(raw), n_ensayo))
            repeticion.append(np.full(len(raw), rep))
            n_ensayo += 1
            time.sleep(2)

    np.savez(RUTA_DATOS, datos=np.concatenate(datos), etiqueta=np.concatenate(etiquetas),
             ensayo=np.concatenate(ensayo_id), repeticion=np.concatenate(repeticion),
             clases=np.array(clases))
    print(f"\nGrabación guardada en {RUTA_DATOS}")


def entrenar(stds_fijas=None):
    d = np.load(RUTA_DATOS)
    clases = [str(c) for c in d["clases"]]
    ensayos = []
    for e in np.unique(d["ensayo"]):
        sel = d["ensayo"] == e
        raw = d["datos"][sel][int(DESCARTAR_S * FS):]
        ensayos.append((raw, int(d["etiqueta"][sel][0]), int(d["repeticion"][sel][0])))

    stds = calcular_stds([r for r, _, _ in ensayos]) if stds_fijas is None else np.asarray(stds_fijas)
    print("Desviaciones estándar nuevas:", np.round(stds, 3))

    # Referencia EA nueva: covarianza media de TODOS los bloques de la calibración,
    # ya centrados y divididos por las stds nuevas (exactamente lo que recibe la EA en tiempo real)
    bloques = [b for raw, _, _ in ensayos for b in bloques_normalizados(raw, stds)]
    EuclideanAlignment().fit(bloques).save(RUTA_EA_NUEVA)
    print(f"Referencia EA nueva calculada con {len(bloques)} bloques -> {RUTA_EA_NUEVA}")

    try:
        from modelo_c import ModeloC
        modelo = ModeloC()
    except Exception:
        modelo = Modelo(RUTA_PESOS)
    ea = OnlineEuclideanAlignment.load(RUTA_EA_NUEVA, prior_count=30, forgetting="keep")

    X, y, rep = [], [], []
    print("Extrayendo características...")
    for i, (raw, etiqueta, r) in enumerate(ensayos):
        v = ventanas_de_ensayo(raw, stds, ea)
        if len(v) == 0:
            continue
        X.append(modelo.caracteristicas(v))
        y.append(np.full(len(v), etiqueta))
        rep.append(np.full(len(v), r))
        print(f"  ensayo {i + 1}/{len(ensayos)}: {len(v)} ventanas")
    X, y, rep = np.concatenate(X), np.concatenate(y), np.concatenate(rep)

    # Validación: última repetición (si hay más de una); si no, 20 % aleatorio
    if rep.max() > 0:
        val = rep == rep.max()
    else:
        val = np.random.default_rng(0).random(len(y)) < 0.2
    Xtr, ytr, Xv, yv = X[~val], y[~val], X[val], y[val]
    print(f"Entrenamiento: {len(ytr)} ventanas, validación: {len(yv)}")

    # Referencia: modelo original (con las stds nuevas) en validación
    n_salidas_original = modelo.ultima_capa(X[:1]).shape[1]
    if len(clases) == n_salidas_original:
        base = np.mean(modelo.ultima_capa(Xv).argmax(1) == yv)
        print(f"Exactitud del modelo original en validación: {base:.3f}")

    mlp = MLP(X.shape[1], N_OCULTA, len(clases))
    mlp.entrenar(Xtr, ytr, Xv, yv)
    acc = np.mean(mlp.predecir(Xv).argmax(1) == yv)
    print(f"Exactitud de la MLP en validación: {acc:.3f}")

    # Matriz de confusión
    cm = np.zeros((len(clases), len(clases)), int)
    for real, pred in zip(yv, mlp.predecir(Xv).argmax(1)):
        cm[real, pred] += 1
    print("Matriz de confusión (filas = real, columnas = predicha):")
    print(cm)

    mlp.guardar(RUTA_SALIDA, stds=stds, clases=clases)
    print(f"\nGuardado {RUTA_SALIDA}")


if __name__ == "__main__":
    if "--solo-entrenar" not in sys.argv:
        clases = CLASES
        if clases is None:
            m = Modelo(RUTA_PESOS)
            n_out = m.w[f"{len(m.capas) - 1}_b"].shape[0]
            clases = [f"g{i}" for i in range(n_out)]
        sesion_de_grabacion(clases)
    entrenar()
