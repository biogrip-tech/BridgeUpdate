"""
Inferencia del modelo CNN + BiGRU usando solo NumPy.
Funciona igual en la PC y en la Luckfox (no necesita TensorFlow).
Requiere el archivo pesos.npz generado por exportar_pesos.py.
"""
import json
import time
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


ACT = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(x, 0),
    "tanh": np.tanh,
    "sigmoid": lambda x: 0.5 * (1.0 + np.tanh(0.5 * x)),   # equivalente, sin overflow
    "softmax": _softmax,
}


def conv1d(x, k, b, padding):
    # x: (N, T, Cin)   k: (K, Cin, Cout)
    K = k.shape[0]
    if padding == "same":
        total = K - 1
        izq = total // 2
        x = np.pad(x, ((0, 0), (izq, total - izq), (0, 0)))
    Cin, Cout = k.shape[1], k.shape[2]
    if Cin < 32:
        # Pocos canales de entrada: una sola multiplicación grande (im2col) es mucho más rápida
        v = sliding_window_view(x, K, axis=1).transpose(0, 1, 3, 2)   # (N, T, K, Cin)
        v = v.reshape(x.shape[0], -1, K * Cin)
        return v @ k.reshape(K * Cin, Cout) + b
    T = x.shape[1] - K + 1
    y = np.zeros((x.shape[0], T, Cout), dtype=np.float32) + b
    for j in range(K):
        y += x[:, j:j + T] @ k[j]
    return y


def maxpool1d(x, p):
    N, T, C = x.shape
    T2 = T // p
    return x[:, :T2 * p].reshape(N, T2, p, C).max(axis=2)


def gru(x, k, r, b, cfg, al_reves=False):
    # x: (N, T, C) -> estado final (N, units)
    if al_reves:
        x = x[:, ::-1]
    u = cfg["units"]
    act = ACT[cfg["activation"]]
    rec_act = ACT[cfg["recurrent_activation"]]
    if b.ndim == 2:          # reset_after=True (default en Keras)
        b_in, b_rec = b[0], b[1]
    else:
        b_in, b_rec = b, np.zeros_like(b)
    xp = x @ k + b_in        # (N, T, 3u): [z | r | h]
    h = np.zeros((x.shape[0], u), dtype=np.float32)
    secuencia = []
    for t in range(xp.shape[1]):
        if cfg["reset_after"]:
            rec = h @ r + b_rec
            z = rec_act(xp[:, t, :u] + rec[:, :u])
            rr = rec_act(xp[:, t, u:2 * u] + rec[:, u:2 * u])
            hh = act(xp[:, t, 2 * u:] + rr * rec[:, 2 * u:])
        else:
            rec = h @ r[:, :2 * u]
            z = rec_act(xp[:, t, :u] + rec[:, :u])
            rr = rec_act(xp[:, t, u:2 * u] + rec[:, u:2 * u])
            hh = act(xp[:, t, 2 * u:] + (rr * h) @ r[:, 2 * u:])
        h = z * h + (1 - z) * hh
        secuencia.append(h)
    if cfg.get("return_sequences", False):
        seq = np.stack(secuencia, axis=1)          # (N, T, u)
        return seq[:, ::-1] if al_reves else seq   # Keras realinea la dirección inversa
    return h


def bigru(x, kf, rf, bf, kb, rb, bb, cfg):
    """GRU bidireccional (reset_after=True) con ambas direcciones en el mismo bucle."""
    u = cfg["units"]
    act = ACT[cfg["activation"]]
    rec_act = ACT[cfg["recurrent_activation"]]
    xs = np.stack([x, x[:, ::-1]])                       # (2, N, T, C)
    k = np.stack([kf, kb])[:, None]                      # (2, 1, C, 3u)
    xp = xs @ k + np.stack([bf[0], bb[0]])[:, None, None, :]   # (2, N, T, 3u)
    r = np.stack([rf, rb])                               # (2, u, 3u)
    b_rec = np.stack([bf[1], bb[1]])[:, None, :]         # (2, 1, 3u)
    h = np.zeros((2, x.shape[0], u), dtype=np.float32)
    secuencia = []
    for t in range(xp.shape[2]):
        rec = h @ r + b_rec                              # (2, N, 3u)
        zr = rec_act(xp[:, :, t, :2 * u] + rec[:, :, :2 * u])
        z, rr = zr[..., :u], zr[..., u:]
        hh = act(xp[:, :, t, 2 * u:] + rr * rec[:, :, 2 * u:])
        h = z * h + (1 - z) * hh
        secuencia.append(h)
    if cfg.get("return_sequences", False):
        seq = np.stack(secuencia, axis=2)                   # (2, N, T, u)
        return np.concatenate([seq[0], seq[1][:, ::-1]], axis=-1)   # (N, T, 2u)
    return np.concatenate([h[0], h[1]], axis=-1)


class Modelo:
    def __init__(self, ruta="pesos.npz"):
        datos = np.load(ruta)
        self.capas = json.loads(str(datos["__arquitectura__"]))
        self.w = {k: datos[k].astype(np.float32) for k in datos.files
                  if k != "__arquitectura__"}
        # BatchNorm precalculado como x * escala + desplazamiento
        for i, c in enumerate(self.capas):
            if c["tipo"] == "BatchNormalization":
                n = self.w[f"{i}_mean"].shape[0]
                g = self.w.get(f"{i}_gamma", np.ones(n, np.float32))
                be = self.w.get(f"{i}_beta", np.zeros(n, np.float32))
                esc = g / np.sqrt(self.w[f"{i}_var"] + c["epsilon"])
                self.w[f"{i}_esc"] = esc.astype(np.float32)
                self.w[f"{i}_desp"] = (be - self.w[f"{i}_mean"] * esc).astype(np.float32)

    def _batch(self, x, tiempos=None, hasta=None):
        x = np.asarray(x, dtype=np.float32)
        capas = self.capas if hasta is None else self.capas[:hasta]
        for i, c in enumerate(capas):
            t0 = time.perf_counter()
            t = c["tipo"]
            W = lambda n: self.w[f"{i}_{n}"]
            if t == "Conv1D":
                x = ACT[c["activation"]](conv1d(x, W("k"), W("b"), c["padding"]))
            elif t == "BatchNormalization":
                x = x * W("esc") + W("desp")
            elif t == "MaxPooling1D":
                x = maxpool1d(x, c["pool_size"])
            elif t == "Bidirectional":
                if x.ndim != 3:
                    raise ValueError(f"La capa {i} (BiGRU) necesita una secuencia (N, T, C) y recibió "
                                     f"{x.shape}. ¿La GRU anterior tiene return_sequences=True?")
                if c["gru"]["reset_after"]:
                    x = bigru(x, W("fwd_k"), W("fwd_r"), W("fwd_b"),
                              W("bwd_k"), W("bwd_r"), W("bwd_b"), c["gru"])
                else:
                    f = gru(x, W("fwd_k"), W("fwd_r"), W("fwd_b"), c["gru"])
                    bk = gru(x, W("bwd_k"), W("bwd_r"), W("bwd_b"), c["gru"], al_reves=True)
                    x = np.concatenate([f, bk], axis=-1)
            elif t == "GRU":
                x = gru(x, W("k"), W("r"), W("b"), c["gru"])
            elif t == "Dense":
                x = ACT[c["activation"]](x @ W("k") + W("b"))
            elif t in ("Dropout", "InputLayer"):
                pass
            else:
                raise ValueError(f"Capa no soportada: {t}")
            if tiempos is not None:
                tiempos[f"{i:2d} {t}"] = tiempos.get(f"{i:2d} {t}", 0) + time.perf_counter() - t0
        return x

    def predecir(self, x):
        """x: (100, 8) para una muestra o (N, 100, 8) para un batch completo."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            return self._batch(x[None])[0]
        return self._batch(x)

    def caracteristicas(self, x):
        """Salida de la penúltima capa (antes del softmax). x: (N, 100, 8)."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        return self._batch(x, hasta=len(self.capas) - 1)

    def ultima_capa(self, feats):
        """Aplica solo la última capa del modelo original sobre las características."""
        i = len(self.capas) - 1
        c = self.capas[i]
        return ACT[c["activation"]](feats @ self.w[f"{i}_k"] + self.w[f"{i}_b"])


if __name__ == "__main__":
    modelo = Modelo("pesos.npz")
    x = np.random.randn(10, 100, 8).astype(np.float32)   # batch de 10 ventanas
    modelo.predecir(x)  # calentamiento
    t0 = time.time()
    n = 20
    for _ in range(n):
        p = modelo.predecir(x)
    print(f"{(time.time() - t0) / n * 1000:.1f} ms por batch de 10 ventanas")
    print("Clases:", p.argmax(axis=1))

    if "--perfil" in sys.argv:
        tiempos = {}
        for _ in range(n):
            modelo._batch(x, tiempos)
        total = sum(tiempos.values())
        print("\nTiempo por capa (promedio por batch):")
        for nombre, tt in tiempos.items():
            print(f"  {nombre:<22} {tt / n * 1000:7.1f} ms  ({tt / total * 100:4.1f}%)")