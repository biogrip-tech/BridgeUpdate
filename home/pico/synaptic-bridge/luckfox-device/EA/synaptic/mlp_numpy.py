"""
MLP pequeña (entrada -> oculta -> clases) entrenada con NumPy puro.
Se usa sobre las características de la penúltima capa del CNN-GRU.
"""
import json
import numpy as np


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, n_entrada, n_oculta, n_clases, semilla=0):
        rng = np.random.default_rng(semilla)
        self.W1 = (rng.standard_normal((n_entrada, n_oculta)) * np.sqrt(2.0 / n_entrada)).astype(np.float32)
        self.b1 = np.zeros(n_oculta, np.float32)
        self.W2 = (rng.standard_normal((n_oculta, n_clases)) * np.sqrt(1.0 / n_oculta)).astype(np.float32)
        self.b2 = np.zeros(n_clases, np.float32)

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        h = np.maximum(z1, 0)
        return z1, h, _softmax(h @ self.W2 + self.b2)

    def predecir(self, X):
        return self._forward(np.asarray(X, np.float32))[2]

    def _perdida(self, X, y, l2):
        p = self.predecir(X)
        ce = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-9))
        return ce + 0.5 * l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))

    def entrenar(self, X, y, X_val=None, y_val=None, epocas=300, lr=1e-3,
                 l2=1e-4, batch=64, paciencia=30, semilla=0, verbose=True):
        X = np.asarray(X, np.float32)
        y = np.asarray(y, int)
        rng = np.random.default_rng(semilla)
        n_clases = self.W2.shape[1]
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1_, b2_, eps, paso = 0.9, 0.999, 1e-8, 0
        hay_val = X_val is not None and len(X_val) > 0
        mejor, mejor_params, sin_mejora = np.inf, None, 0

        for ep in range(epocas):
            orden = rng.permutation(len(X))
            for ini in range(0, len(X), batch):
                idx = orden[ini:ini + batch]
                xb, yb = X[idx], y[idx]
                z1, h, p = self._forward(xb)
                d2 = p.copy()
                d2[np.arange(len(yb)), yb] -= 1
                d2 /= len(yb)
                gW2 = h.T @ d2 + l2 * self.W2
                gb2 = d2.sum(0)
                d1 = (d2 @ self.W2.T) * (z1 > 0)
                gW1 = xb.T @ d1 + l2 * self.W1
                gb1 = d1.sum(0)
                paso += 1
                for i, g in enumerate([gW1, gb1, gW2, gb2]):
                    m[i] = b1_ * m[i] + (1 - b1_) * g
                    v[i] = b2_ * v[i] + (1 - b2_) * g * g
                    mh = m[i] / (1 - b1_ ** paso)
                    vh = v[i] / (1 - b2_ ** paso)
                    params[i] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

            if hay_val:
                perd = self._perdida(X_val, y_val, l2)
                if perd < mejor - 1e-4:
                    mejor, sin_mejora = perd, 0
                    mejor_params = [p.copy() for p in params]
                else:
                    sin_mejora += 1
                if verbose and (ep % 20 == 0 or sin_mejora == 0 and ep < 5):
                    acc = np.mean(self.predecir(X_val).argmax(1) == y_val)
                    print(f"  época {ep:3d}  pérdida val {perd:.4f}  exactitud val {acc:.3f}")
                if sin_mejora >= paciencia:
                    if verbose:
                        print(f"  Parada temprana en época {ep}")
                    break
            elif verbose and ep % 20 == 0:
                print(f"  época {ep:3d}  pérdida {self._perdida(X, y, l2):.4f}")

        if mejor_params is not None:
            for p, mp in zip(params, mejor_params):
                p[...] = mp
        return self

    def guardar(self, ruta, **extra):
        np.savez(ruta, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 __extra__=np.array(json.dumps({k: np.asarray(v).tolist() for k, v in extra.items()})))

    @classmethod
    def cargar(cls, ruta):
        d = np.load(ruta)
        mlp = cls.__new__(cls)
        mlp.W1, mlp.b1, mlp.W2, mlp.b2 = (d[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2"))
        extra = json.loads(str(d["__extra__"])) if "__extra__" in d.files else {}
        return mlp, extra
