"""
Euclidean Alignment (EA) for EMG/EEG preprocessing.

Reference:
    He, H., & Wu, D. (2020). "Transfer Learning for Brain-Computer Interfaces:
    A Euclidean Space Data Alignment Approach." IEEE Trans. Biomed. Eng.

For a set of trials {X_i} from one domain (subject/session), compute the mean
of their spatial covariance matrices, R_bar, then apply R_bar^{-1/2} to every
trial. After alignment the domain's mean covariance is the identity, so
different domains line up -- useful before cross-subject/session transfer.

Data convention: each trial is (samples, channels). Trials may differ in
length; the channel count must match. Inputs to fit/transform may be a single
2D trial, a 3D array (n_trials, samples, channels), or a list of 2D trials.

Classes:
    EuclideanAlignment        -- batch: fit on a set, then transform.
    OnlineEuclideanAlignment  -- incremental: align trials as they arrive,
                                 optionally warm-started from a saved reference.

Persistence: both classes share an .npz schema (R_bar, count, n_channels,
trace_norm, forgetting, eps), so a reference saved by one can be loaded by the
other. Fit offline with EuclideanAlignment, `.save(...)`, then load it into
OnlineEuclideanAlignment to warm-start a live session.
"""

import numpy as np


def _inv_sqrt_psd(R, eps=1e-10):
    """Inverse matrix square-root of a symmetric PSD matrix via eigendecomp."""
    w, V = np.linalg.eigh(R)
    w = np.clip(w, eps, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def _save_npz(path, R_bar, count, n_channels, trace_norm, forgetting, eps):
    np.savez(
        path,
        R_bar=np.asarray(R_bar, dtype=float),
        count=np.int64(count),
        n_channels=np.int64(n_channels),
        trace_norm=bool(trace_norm),
        forgetting=(np.nan if forgetting is None else float(forgetting)),
        eps=float(eps),
    )


def _load_npz(path):
    d = np.load(path)
    f = float(d["forgetting"])
    return {
        "R_bar": d["R_bar"],
        "count": int(d["count"]),
        "n_channels": int(d["n_channels"]),
        "trace_norm": bool(d["trace_norm"]),
        "forgetting": (None if np.isnan(f) else f),
        "eps": float(d["eps"]),
    }


class EuclideanAlignment:
    """Batch EA: fit the reference on a set of trials, then transform."""

    def __init__(self, trace_norm=False, eps=1e-10):
        self.trace_norm = trace_norm
        self.eps = eps
        self.R_bar_ = None
        self.ref_inv_sqrt_ = None
        self.n_channels_ = None
        self.n_trials_ = 0

    def _to_list(self, X):
        if isinstance(X, (list, tuple)):
            trials = [np.asarray(t, dtype=float) for t in X]
            if any(t.ndim != 2 for t in trials):
                raise ValueError("Each trial in a list must be 2D (samples, channels).")
            return trials, "list"
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            return [X], "single"
        if X.ndim == 3:
            return list(X), "array"
        raise ValueError("Expected 2D trial, 3D array, or list of 2D trials.")

    def _cov(self, trial):
        n = trial.shape[0]
        R = (trial.T @ trial) / n
        if self.trace_norm:
            tr = np.trace(R)
            if tr > 0:
                R = R / tr
        return R

    def fit(self, X):
        trials, _ = self._to_list(X)
        n_ch = trials[0].shape[1]
        if any(t.shape[1] != n_ch for t in trials):
            raise ValueError("All trials must have the same number of channels.")
        self.n_channels_ = n_ch
        self.n_trials_ = len(trials)
        self.R_bar_ = np.mean([self._cov(t) for t in trials], axis=0)
        self.ref_inv_sqrt_ = _inv_sqrt_psd(self.R_bar_, self.eps)
        return self

    def transform(self, X):
        if self.ref_inv_sqrt_ is None:
            raise RuntimeError("Call fit() (or load()) before transform().")
        trials, kind = self._to_list(X)
        if any(t.shape[1] != self.n_channels_ for t in trials):
            raise ValueError("Channel count differs from the fitted reference.")
        aligned = [t @ self.ref_inv_sqrt_ for t in trials]
        if kind == "single":
            return aligned[0]
        if kind == "array":
            return np.stack(aligned)
        return aligned

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # --- persistence / handoff -------------------------------------------
    def save(self, path):
        """Save the reference to an .npz file (use a path ending in .npz)."""
        if self.R_bar_ is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        _save_npz(path, self.R_bar_, self.n_trials_, self.n_channels_,
                  self.trace_norm, None, self.eps)
        return self

    @classmethod
    def load(cls, path):
        """Load a saved reference for fixed (non-adapting) alignment."""
        s = _load_npz(path)
        obj = cls(trace_norm=s["trace_norm"], eps=s["eps"])
        obj.R_bar_ = s["R_bar"]
        obj.n_channels_ = s["n_channels"]
        obj.n_trials_ = s["count"]
        obj.ref_inv_sqrt_ = _inv_sqrt_psd(obj.R_bar_, obj.eps)
        return obj

    def to_online(self, prior_count=1, forgetting=None, trace_norm=None):
        """
        Hand this fitted reference to an OnlineEuclideanAlignment as a warm
        start. `prior_count` is the pseudo-count the saved reference is worth
        (how strongly it is anchored against incoming trials).
        """
        if self.R_bar_ is None:
            raise RuntimeError("Fit first before handing off to online.")
        on = OnlineEuclideanAlignment(
            forgetting=forgetting,
            trace_norm=self.trace_norm if trace_norm is None else trace_norm,
            eps=self.eps,
        )
        return on._seed(self.R_bar_, prior_count, self.n_channels_)


class OnlineEuclideanAlignment:
    """
    Incremental EA. Align trials as they arrive, optionally warm-started from a
    saved reference so the first iterations are already aligned.

    Parameters
    ----------
    forgetting : float or None
        None      -> cumulative running mean (converges for a stationary
                     session; a warm-start prior washes out as trials
                     accumulate, at a rate set by its pseudo-count).
        f in (0,1)-> exponential update R_bar <- f*R_bar + (1-f)*R_new; a
                     warm-start prior governs the first ~1/(1-f) trials, then
                     the current session takes over. Use ~0.9-0.99 for drift.
    trace_norm, eps : see EuclideanAlignment.

    

    buf, W, hop = [], 100, 50          # 200 ms window, 100 ms hop at 500 sps
    ea = OnlineEuclideanAlignment.load("subj01_ref.npz", prior_count=30)
    while streaming:
        buf.extend(read_buffer())
        while len(buf) >= W:
            window = np.array(buf[:W]); del buf[:hop]
            aligned = ea.partial_fit_transform(window)   # per-window EA, warm-started
            # -> features / model
            
    """

    def __init__(self, forgetting=None, trace_norm=False, eps=1e-10):
        if forgetting is not None and not (0.0 < forgetting < 1.0):
            raise ValueError("forgetting must be in (0, 1) or None.")
        self.forgetting = forgetting
        self.trace_norm = trace_norm
        self.eps = eps
        self.R_bar_ = None
        self.ref_inv_sqrt_ = None
        self.n_channels_ = None
        self.k_ = 0

    def _cov(self, trial):
        trial = np.asarray(trial, dtype=float)
        if trial.ndim != 2:
            raise ValueError("A trial must be 2D (samples, channels).")
        n = trial.shape[0]
        R = (trial.T @ trial) / n
        if self.trace_norm:
            tr = np.trace(R)
            if tr > 0:
                R = R / tr
        return R

    def _seed(self, R_bar, count, n_channels=None):
        """Warm-start the running reference from a prior covariance."""
        self.R_bar_ = np.array(R_bar, dtype=float, copy=True)
        self.k_ = int(count)
        self.n_channels_ = n_channels if n_channels is not None else self.R_bar_.shape[0]
        self.ref_inv_sqrt_ = _inv_sqrt_psd(self.R_bar_, self.eps)
        return self

    def partial_fit(self, trial):
        R = self._cov(trial)
        if self.n_channels_ is None:
            self.n_channels_ = R.shape[0]
        elif R.shape[0] != self.n_channels_:
            raise ValueError("Channel count changed between trials.")
        self.k_ += 1
        if self.R_bar_ is None:
            self.R_bar_ = R.copy()
        elif self.forgetting is None:
            self.R_bar_ = self.R_bar_ + (R - self.R_bar_) / self.k_
        else:
            f = self.forgetting
            self.R_bar_ = f * self.R_bar_ + (1.0 - f) * R
        self.ref_inv_sqrt_ = _inv_sqrt_psd(self.R_bar_, self.eps)
        return self

    def transform(self, trial):
        if self.ref_inv_sqrt_ is None:
            raise RuntimeError("No reference yet; seed/load or partial_fit first.")
        return np.asarray(trial, dtype=float) @ self.ref_inv_sqrt_

    def partial_fit_transform(self, trial, update_first=True):
        if update_first:
            self.partial_fit(trial)
            return self.transform(trial)
        out = self.transform(trial) if self.ref_inv_sqrt_ is not None \
            else np.asarray(trial, dtype=float)
        self.partial_fit(trial)
        return out

    # --- persistence / warm start ----------------------------------------
    @classmethod
    def from_reference(cls, R_bar, count=1, forgetting=None,
                       trace_norm=False, eps=1e-10):
        """Build a warm-started online aligner from a prior covariance."""
        return cls(forgetting=forgetting, trace_norm=trace_norm,
                   eps=eps)._seed(R_bar, count)

    def save(self, path):
        if self.R_bar_ is None:
            raise RuntimeError("Nothing to save; no reference yet.")
        _save_npz(path, self.R_bar_, self.k_, self.n_channels_,
                  self.trace_norm, self.forgetting, self.eps)
        return self

    @classmethod
    def load(cls, path, prior_count=None, forgetting="keep"):
        """
        Load a saved reference as a warm start.

        prior_count : override the pseudo-count the prior is worth (defaults to
                      the count stored in the file). Small -> prior washes out
                      fast; large -> prior stays anchored.
        forgetting  : "keep" uses the saved value; pass None or a float to
                      override (e.g. set a forgetting factor for a session you
                      expect to drift from the saved reference).
        """
        s = _load_npz(path)
        f = s["forgetting"] if forgetting == "keep" else forgetting
        count = s["count"] if prior_count is None else prior_count
        obj = cls(forgetting=f, trace_norm=s["trace_norm"], eps=s["eps"])
        return obj._seed(s["R_bar"], count, s["n_channels"])


class StreamingWhitener:
    """
    Continuous adaptive spatial whitening for a real-time sample stream.

    Maintains an exponentially-weighted running covariance C over incoming
    samples and applies C^{-1/2} to each chunk, so the running spatial
    covariance of the output stays near the identity. Suited to continuous
    (non-epoched) EMG. The whitening matrix is refreshed every `refresh_every`
    samples (cheap covariance update every chunk; the O(channels^3)
    eigendecomposition only on refresh).

    Choosing the memory:
        Give `fs` and `time_constant` (seconds); the per-sample forgetting is
        lambda = exp(-1 / (fs * time_constant)). At fs=500, time_constant=2.0
        -> lambda ~ 0.999 (~2 s of memory). Longer = steadier but slower to
        follow drift; shorter = adapts faster, noisier.

    Warm start:
        seed(C) or load(path) with a prior covariance (e.g. EA's saved R_bar --
        it IS a covariance, so the same .npz works) so the first samples are
        already whitened instead of passing through near-identity.

    Notes:
        - No centering by default (matches EA's X^T X convention; EMG is ~zero
          mean after band-pass). Set subtract_mean=True to track/remove DC.
        - `shrinkage` mixes C toward a scaled identity for stability when
          channels are strongly correlated or near-singular (common in EMG).
        - Whitening is a matrix multiply -> negligible latency; the refresh is
          on a small channels x channels matrix -> fine at 500 sps.
    """

    def __init__(self, fs=None, time_constant=2.0, forgetting=None,
                 refresh_every=25, shrinkage=1e-3, subtract_mean=False,
                 eps=1e-10):
        if forgetting is not None:
            if not (0.0 < forgetting < 1.0):
                raise ValueError("forgetting must be in (0, 1).")
            self.lambda_ = float(forgetting)
        elif fs is not None:
            self.lambda_ = float(np.exp(-1.0 / (fs * time_constant)))
        else:
            raise ValueError("Provide `forgetting`, or `fs` (with time_constant).")
        self.fs = fs
        self.refresh_every = int(refresh_every)
        self.shrinkage = float(shrinkage)
        self.subtract_mean = subtract_mean
        self.eps = eps
        self.C_ = None
        self.W_ = None
        self.mean_ = None
        self.n_channels_ = None
        self.n_seen_ = 0
        self._since_refresh = 0

    @property
    def memory_samples(self):
        return 1.0 / (1.0 - self.lambda_)

    @property
    def memory_seconds(self):
        return self.memory_samples / self.fs if self.fs else None

    def _ensure(self, n_channels):
        if self.C_ is None:
            self.n_channels_ = n_channels
            self.C_ = np.eye(n_channels)      # neutral start -> W = I (passthrough)
            self.mean_ = np.zeros(n_channels)
            self.W_ = np.eye(n_channels)

    def _refresh(self):
        n = self.C_.shape[0]
        tr = np.trace(self.C_) / n
        C_reg = (1.0 - self.shrinkage) * self.C_ + self.shrinkage * tr * np.eye(n)
        self.W_ = _inv_sqrt_psd(C_reg, self.eps)
        self._since_refresh = 0

    def seed(self, C, n_seen=0):
        """Warm-start the running covariance from a prior (e.g. EA's R_bar)."""
        C = np.asarray(C, dtype=float)
        self.n_channels_ = C.shape[0]
        self.C_ = C.copy()
        self.mean_ = np.zeros(self.n_channels_)
        self.n_seen_ = int(n_seen)
        self._refresh()
        return self

    def _update(self, chunk):
        m = chunk.shape[0]
        lam_m = self.lambda_ ** m                     # block forgetting over the chunk
        if self.subtract_mean:
            self.mean_ = lam_m * self.mean_ + (1.0 - lam_m) * chunk.mean(axis=0)
            chunk = chunk - self.mean_
        Rc = (chunk.T @ chunk) / m
        self.C_ = lam_m * self.C_ + (1.0 - lam_m) * Rc
        self.n_seen_ += m
        self._since_refresh += m

    def partial_fit_transform(self, chunk):
        """Update the running covariance from `chunk` and return it whitened.

        `chunk` is (n_samples, channels); a single sample (channels,) is fine.
        Keep chunks small relative to the memory (<< memory_samples).
        """
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        self._ensure(chunk.shape[1])
        self._update(chunk)
        if self._since_refresh >= self.refresh_every:
            self._refresh()
        out = (chunk - self.mean_) if self.subtract_mean else chunk
        return out @ self.W_

    def transform(self, chunk):
        """Whiten with the current matrix without updating (no adaptation)."""
        if self.W_ is None:
            raise RuntimeError("No whitening matrix yet; seed/load or feed data first.")
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        out = (chunk - self.mean_) if self.subtract_mean else chunk
        return out @ self.W_

    @classmethod
    def from_reference(cls, C, **kwargs):
        return cls(**kwargs).seed(C)

    @classmethod
    def load(cls, path, **kwargs):
        """Warm-start from a saved EA/streaming .npz (uses its covariance)."""
        s = _load_npz(path)
        return cls(**kwargs).seed(s["R_bar"])