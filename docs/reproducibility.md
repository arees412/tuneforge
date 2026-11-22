# Reproducibility

TuneForge records source hashes, canonical dataset fingerprints, exact split membership, model configuration and revision, adapter settings, training config, plan hash, package version, seed, runtime capabilities, measured metrics, and checkpoint hashes.

A single bounded byte buffer supplies both the source hash and the decoded records. Each run persists its dataset manifest. The evidence bundle carries the dataset fingerprint and the digest of that manifest. Reopening a modified source requires a new registration. A weights-only restart records its parent and uses a fresh optimizer, so a resumed trajectory can differ from uninterrupted training.

The local backend seeds Python and PyTorch and enables deterministic PyTorch algorithms. Stable JSON uses sorted keys and normalized timestamps/paths. Evidence ZIP members are sorted and receive fixed timestamps and permissions so identical evidence inputs produce identical archive bytes.

Reproducibility is bounded. PyTorch does not guarantee identical results across versions, platforms, devices, or all kernels. GPU reductions, third-party quantization kernels, dependency changes, and thread scheduling can alter floating-point results. The manifest therefore describes the environment instead of promising universal bitwise equality.

CI fixes Python and locks package versions in `uv.lock`, uses only local generated fixtures, constructs model weights from configuration, and requires neither GPU nor service credentials.
