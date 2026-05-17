# NanoMoSeq

Minimal AR-HMM pipeline for unsupervised behavioral segmentation from pose keypoints.

## Installation

```bash
pip install nanomoseq
```

## Pipeline

```
keypoints (T, K, 3)
       │
       ▼
  NanoMoseq          egocentric alignment → PCA → AR-HMM Gibbs
       │
       ▼
 syllable labels (T,)
       │
       ▼
  NanoHHMM           hierarchical HMM Gibbs
       │
       ▼
 behavioral states (T,)
```

## Usage

```python
from nanomoseq import NanoMoseq, NanoHHMM

# Keypoints: dict of recordings, each array of shape (T, K, 3)
#   T = number of frames
#   K = number of keypoints
#   3 = [x, y, confidence]

keypoints = {
    "mouse_01": ...,  # (T, K, 3)
    "mouse_02": ...,  # (T, K, 3)
}

# Stage 1 — segment keypoint sequences into syllables
moseq = NanoMoseq(
    n_syllables=10,   # number of behavioral syllables
    n_dims=10,        # PCA dimensions to retain
    n_iters=100,      # Gibbs iterations
    kappa=100.0,      # self-transition bias
    alpha=1.0,        # Dirichlet concentration
)
moseq.fit(keypoints)

syllable_labels = moseq.labels_            # {name: ndarray(T,)}
pca_coords      = moseq.pca_coords_        # {name: ndarray(T, n_dims)}
transition_mat  = moseq.transition_matrix_ # (n_syllables, n_syllables)

# Stage 2 — group syllable sequences into higher-level behavioral states
hhmm = NanoHHMM(
    n_states=5,           # number of behavioral states
    n_iters=100,
    kappa=100.0,
    alpha=1.0,
    emissions_beta=0.1,   # Dirichlet prior on syllable emissions
)
hhmm.fit(moseq.labels_)

behavioral_states = hhmm.states_      # {name: ndarray(T,)}
trans_probs       = hhmm.trans_probs_ # (n_states, n_states)
emissions         = hhmm.emissions_   # (n_states, n_syllables, n_syllables)
```

## Citation

If you use NanoMoSeq in your research, please cite it as:

```bibtex
@software{cobos2026nanomoseq,
  author  = {Cobos, Juan},
  title   = {{NanoMoSeq}: Minimal AR-HMM pipeline for unsupervised behavioral segmentation from pose keypoints},
  year    = {2026},
  url     = {https://github.com/juan-cobos/nanomoseq},
  version = {0.1.0},
  license = {MIT}
}
```

A machine-readable citation is also available in [CITATION.cff](CITATION.cff).

## Acknowledgments

- Weinreb, C., Pearl, J. E., Lin, S., Osman, M. A. M., Zhang, L., Annapragada, S., Conlin, E., Hoffmann, R., Makowska, S., Gillis, W. F., Jay, M., Ye, S., Mathis, A., Mathis, M. W., Pereira, T., Linderman, S. W., & Datta, S. R. (2024). Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics. *Nature Methods*, 21(7), 1329–1339. https://doi.org/10.1038/s41592-024-02318-2

- Weinreb, C., Kannan, L. T., Newman-Boulle, A., Sainburg, T., Gillis, W. F., Plotnikoff, A., Makowska, S., Pearl, J. E., Osman, M. A. M., Linderman, S. W., & Datta, S. R. (2026). Spontaneous behavior is a succession of self-directed tasks. *Neuron*, 114(5), 922–937.e12. https://doi.org/10.1016/j.neuron.2025.11.021
