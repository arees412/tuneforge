# Checkpoints

Each checkpoint manifest records its ID, owning run, step, epoch, time, method, base reference, adapter/full kind, relative artifact paths, SHA-256 hashes, measured metrics, model-manifest hash, and training-configuration digest.

Artifacts live under `artifacts/runs/<run-id>/checkpoints/step-<n>`. Absolute paths and traversal are rejected. Existing checkpoint directories are not overwritten. Retention keeps the newest configured count plus manual checkpoints and removes only directories already proven to belong to that run.

Model or adapter weights use safetensors through `save_pretrained`. Full PyTorch optimizer-state serialization is intentionally absent; TuneForge does not automatically load untrusted pickle artifacts.

Resume recomputes every artifact hash and requires exact model-manifest and training-config compatibility. A mismatch or tampered file fails closed.

