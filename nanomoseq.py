"""
NanoMoSeq — single-file AR-HMM behavioral segmentation pipeline (numpy only).

Pipeline
--------
  Stage 1 — NanoMoseq
      Egocentric alignment  : confidence-weighted centroid + PCA heading
      PCA                   : K*2 keypoint dims → n_dims
      Gibbs sampler         : MNIW AR params + Dirichlet transitions + FFBS
      Output                : syllable label sequence per recording

  Stage 2 — NanoHHMM
      Hierarchical HMM over syllable sequences
      Gibbs sampler         : Dirichlet emissions + transitions + FFBS
      Output                : behavioral state sequence per recording
"""

import numpy as np

# ── Utilities ─────────────────────────────────────────────────────────────────


def _logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return out.squeeze(axis=axis) if axis is not None else out.squeeze()


def _sample_invwishart(Psi, nu):
    """Sample Q ~ IW(Psi, nu) via Bartlett decomposition."""
    d = Psi.shape[0]
    L = np.linalg.cholesky(np.linalg.inv(Psi))
    A = np.zeros((d, d))
    for i in range(d):
        A[i, i] = np.sqrt(np.random.chisquare(nu - i))
        A[i, :i] = np.random.randn(i)
    X = L @ A
    return np.linalg.inv(X @ X.T + 1e-9 * np.eye(d))


def _ffbs(log_em_seq, log_pi, log_init):
    """Forward-filtering backward-sampling.

    log_em_seq : (T-1, n_states)  — pre-built log emission for each transition
    log_pi     : (n_states, n_states)
    log_init   : (n_states,)
    Returns    : (T,) int state sequence
    """
    T1, S = log_em_seq.shape
    T = T1 + 1

    log_a = np.zeros((T, S))
    log_a[0] = log_init
    for t in range(1, T):
        log_a[t] = log_em_seq[t - 1] + _logsumexp(
            log_a[t - 1, :, None] + log_pi, axis=0
        )
        log_a[t] -= _logsumexp(log_a[t])

    z = np.empty(T, dtype=int)
    lp = log_a[-1] - _logsumexp(log_a[-1])
    z[-1] = np.random.choice(S, p=np.exp(lp))
    for t in range(T - 2, -1, -1):
        lp = log_a[t] + log_pi[:, z[t + 1]]
        lp -= _logsumexp(lp)
        z[t] = np.random.choice(S, p=np.exp(lp))

    return z


# ── Stage 1: AR-HMM syllable segmentation ────────────────────────────────────


class NanoMoseq:
    def __init__(
        self,
        n_syllables=10,
        n_dims=10,
        n_iters=100,
        kappa=100.0,
        alpha=1.0,
    ):
        self.n_syllables = n_syllables
        self.n_dims = n_dims
        self.n_iters = n_iters
        self.kappa = kappa
        self.alpha = alpha

    def fit(self, keypoints_dict):
        """
        Parameters
        ----------
        keypoints_dict : {name: ndarray(T, K, 3)}
            Last dimension is [x, y, confidence].
        """
        self.names_ = list(keypoints_dict)

        ego = [
            self._align_egocentric(
                keypoints_dict[n][..., :2], keypoints_dict[n][..., 2]
            )
            for n in self.names_
        ]

        self._fit_pca(ego)

        sequences = [self._project(e) for e in ego]
        self.pca_coords_ = dict(zip(self.names_, sequences))

        labels = self._gibbs(sequences)
        self.labels_ = dict(zip(self.names_, labels))

        return self

    def _align_egocentric(self, coords, confs):
        T, K, _ = coords.shape
        out = np.zeros_like(coords)
        prev_heading = None

        for t in range(T):
            w = confs[t] + 1e-9
            w /= w.sum()
            center = (w[:, None] * coords[t]).sum(0)
            pts = coords[t] - center

            # first eigenvector of weighted covariance = dominant body axis
            cov = pts.T @ np.diag(w) @ pts
            _, eigvecs = np.linalg.eigh(cov)
            heading = eigvecs[:, -1]

            # resolve 180° sign ambiguity by continuity
            if prev_heading is not None and heading @ prev_heading < 0:
                heading = -heading
            prev_heading = heading

            angle = np.arctan2(heading[1], heading[0])
            ca, sa = np.cos(-angle), np.sin(-angle)
            out[t] = pts @ np.array([[ca, sa], [-sa, ca]])

        return out

    def _fit_pca(self, sequences):
        X = np.concatenate([s.reshape(len(s), -1) for s in sequences])
        self._pca_mean = X.mean(0)
        X -= self._pca_mean
        _, s_vals, Vt = np.linalg.svd(X, full_matrices=False)
        self._pca_components = Vt[: self.n_dims]
        var = (s_vals[: self.n_dims] ** 2).sum() / (s_vals**2).sum()
        print(f"PCA: {self.n_dims} PCs → {var:.1%} of variance")

    def _project(self, seq):
        return (seq.reshape(len(seq), -1) - self._pca_mean) @ self._pca_components.T

    # ── Gibbs sampler ─────────────────────────────────────────────────────────
    #
    # Model:  x_t = A_{z_t} @ x_{t-1} + ε,   ε ~ N(0, Q_{z_t})
    #
    # Three alternating steps:
    #   (a) AR params (A_s, Q_s) — MNIW conjugate update
    #   (b) transition matrix π  — Dirichlet conjugate
    #   (c) syllable labels z    — forward-filtering backward-sampling

    def _gibbs(self, sequences):
        S, d = self.n_syllables, self.n_dims
        prior = dict(
            M0=np.zeros((d, d)), V0=10.0 * np.eye(d), Psi0=np.eye(d), nu0=float(d + 2)
        )

        As = [np.eye(d) * 0.9 for _ in range(S)]
        Qs = [np.eye(d) for _ in range(S)]
        pi = np.eye(S) * self.kappa + np.ones((S, S)) * self.alpha
        pi /= pi.sum(1, keepdims=True)
        log_init = np.full(S, -np.log(S))
        labels = [np.random.randint(0, S, len(seq)) for seq in sequences]

        for it in range(self.n_iters):
            # (a) AR params — collect (x_{t-1}, x_t) pairs per syllable
            for s in range(S):
                xp, xc = [], []
                for seq, z in zip(sequences, labels):
                    idx = np.where(z == s)[0]
                    idx = idx[idx > 0]
                    if len(idx):
                        xp.append(seq[idx - 1])
                        xc.append(seq[idx])
                xp = np.concatenate(xp) if xp else np.zeros((0, d))
                xc = np.concatenate(xc) if xc else np.zeros((0, d))
                As[s], Qs[s] = self._sample_ar(xp, xc, prior)

            # (b) transition matrix
            counts = np.zeros((S, S))
            for z in labels:
                for t in range(len(z) - 1):
                    counts[z[t], z[t + 1]] += 1
            for s in range(S):
                conc = counts[s] + self.alpha
                conc[s] += self.kappa
                pi[s] = np.random.dirichlet(conc)

            # (c) labels
            log_pi = np.log(pi + 1e-12)
            labels = [self._ffbs(seq, As, Qs, log_pi, log_init) for seq in sequences]

            if (it + 1) % 10 == 0:
                all_z = np.concatenate(labels)
                cnts = np.bincount(all_z, minlength=S)
                print(
                    f"  iter {it + 1:3d}: {(cnts > 0).sum()} active  "
                    f"top-5: {np.sort(cnts)[::-1][:5]}"
                )

        self.transition_matrix_ = pi
        return labels

    @staticmethod
    def _sample_ar(x_prev, x_curr, prior):
        """MNIW conjugate update for (A, Q).

        Likelihood:  x_curr = A @ x_prev + ε,  ε ~ N(0, Q)
        i.e.         Y = A X + E  with Y,X ∈ R^{d×n}
        Prior:       Q ~ IW(Ψ₀,ν₀),  A|Q ~ MN(M₀, Q, V₀)
        """
        M0, V0, Psi0, nu0 = prior["M0"], prior["V0"], prior["Psi0"], prior["nu0"]
        d = M0.shape[0]

        if len(x_prev) == 0:
            Q = _sample_invwishart(Psi0, nu0)
            A = (
                M0
                + np.linalg.cholesky(Q)
                @ np.random.randn(d, d)
                @ np.linalg.cholesky(V0).T
            )
            return A, Q

        X, Y = x_prev.T, x_curr.T
        n = X.shape[1]
        V0inv = np.linalg.inv(V0)

        Vninv = V0inv + X @ X.T
        Mn = np.linalg.solve(Vninv, (M0 @ V0inv + Y @ X.T).T).T
        Psin = Psi0 + Y @ Y.T + M0 @ V0inv @ M0.T - Mn @ Vninv @ Mn.T
        Psin = 0.5 * (Psin + Psin.T) + 1e-6 * np.eye(d)

        Q = _sample_invwishart(Psin, nu0 + n)
        Q = 0.5 * (Q + Q.T) + 1e-6 * np.eye(d)
        Vn = np.linalg.inv(Vninv + 1e-9 * np.eye(d))
        A = (
            Mn
            + np.linalg.cholesky(Q) @ np.random.randn(d, d) @ np.linalg.cholesky(Vn).T
        )
        return A, Q

    @staticmethod
    def _ffbs(x, As, Qs, log_pi, log_init):
        """Build Gaussian AR log-emissions then delegate to shared FFBS."""
        T, d = x.shape

        As_arr = np.stack(As)  # (S, d, d)
        Q_reg = np.stack(Qs) + 1e-9 * np.eye(d)  # (S, d, d)
        Q_inv = np.linalg.inv(Q_reg)
        c = -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(Q_reg)[1])

        preds = np.einsum("sde,te->std", As_arr, x[:-1])
        diffs = x[1:][None] - preds
        log_em_seq = (
            c[:, None] - 0.5 * np.einsum("std,sde,ste->st", diffs, Q_inv, diffs)
        ).T

        return _ffbs(log_em_seq, log_pi, log_init)


# ── Stage 2: Hierarchical HMM over syllable sequences ────────────────────────


class NanoHHMM:
    """Hierarchical HMM over MoSeq syllable sequences.

    Two-level model
    ---------------
      Level 1 (NanoMoseq)  :  keypoint data  →  syllable labels  z_t ∈ {0..S-1}
      Level 2 (this class)   :  syllable labels →  behavioral state q_t ∈ {0..M-1}

    Generative model
    ----------------
      q_t | q_{t-1}         ~ Categorical( trans_probs[q_{t-1}] )
      z_t | z_{t-1}, q_t    ~ Categorical( emissions[q_t, z_{t-1}] )
    """

    def __init__(
        self,
        n_states=5,
        n_iters=100,
        kappa=100.0,
        alpha=1.0,
        emissions_beta=0.1,
    ):
        self.n_states = n_states
        self.n_iters = n_iters
        self.kappa = kappa
        self.alpha = alpha
        self.emissions_beta = emissions_beta

    def fit(self, labels_dict):
        """
        Parameters
        ----------
        labels_dict : {name: ndarray(T,)}
            Integer syllable sequences — e.g. NanoMoseq.labels_.
        """
        self.names_ = list(labels_dict)
        sequences = [labels_dict[n].astype(int) for n in self.names_]
        self.n_syllables_ = max(s.max() for s in sequences) + 1

        state_seqs = self._gibbs(sequences)
        self.states_ = dict(zip(self.names_, state_seqs))
        return self

    # ── Gibbs sampler ─────────────────────────────────────────────────────────
    #
    # (a) emissions[m]   — Dirichlet conjugate update per (state, from-syllable)
    # (b) trans_probs[m] — Dirichlet conjugate with kappa self-transition boost
    # (c) state labels q — forward-filtering backward-sampling over q

    def _gibbs(self, sequences):
        M, S = self.n_states, self.n_syllables_

        # emissions[m, s_from, s_to] = P(z_t = s_to | z_{t-1} = s_from, q_t = m)
        emissions = np.ones((M, S, S)) / S
        pi = np.eye(M) * self.kappa + np.ones((M, M)) * self.alpha
        pi /= pi.sum(1, keepdims=True)
        log_init = np.full(M, -np.log(M))
        states = [np.random.randint(0, M, len(z)) for z in sequences]

        for it in range(self.n_iters):
            # (a) emissions — Dirichlet update per (state, from-syllable)
            counts = np.zeros((M, S, S))
            for z, q in zip(sequences, states):
                np.add.at(counts, (q[1:], z[:-1], z[1:]), 1)
            for m in range(M):
                for s in range(S):
                    conc = counts[m, s] + self.emissions_beta
                    emissions[m, s] = np.random.dirichlet(conc)

            # (b) transition matrix — Dirichlet update
            trans_counts = np.zeros((M, M))
            for q in states:
                np.add.at(trans_counts, (q[:-1], q[1:]), 1)
            for m in range(M):
                conc = trans_counts[m] + self.alpha
                conc[m] += self.kappa
                pi[m] = np.random.dirichlet(conc)

            # (c) state labels — FFBS
            log_pi = np.log(pi + 1e-12)
            log_em = np.log(emissions + 1e-12)  # (M, S, S)
            states = [self._ffbs(z, log_em, log_pi, log_init) for z in sequences]

            if (it + 1) % 10 == 0:
                all_q = np.concatenate(states)
                cnts = np.bincount(all_q, minlength=M)
                print(
                    f"  iter {it + 1:3d}: {(cnts > 0).sum()} active  "
                    f"top counts: {np.sort(cnts)[::-1]}"
                )

        self.emissions_ = emissions
        self.trans_probs_ = pi
        return states

    @staticmethod
    def _ffbs(z, log_em, log_pi, log_init):
        """Build categorical log-emissions then delegate to shared FFBS."""
        log_em_seq = log_em[:, z[:-1], z[1:]].T  # (T-1, M)
        return _ffbs(log_em_seq, log_pi, log_init)
