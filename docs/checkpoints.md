# Checkpoints

Each checkpoint belongs to one run and step. Its manifest binds dataset fingerprint, model manifest, training configuration, weight format, relative filenames, and SHA-256 digests. The owning run and active checkpoint registration live in SQLite. Full models and LoRA adapters use safetensors.

## Loading

CLI evaluation, weights-only resume, and the trainer's reload check use the same loader. It verifies ownership, the on-disk manifest, training state, and the complete expected set of regular files. It copies the listed members into a private directory, verifies those bytes, and loads locally from that directory. Missing or unexpected files, changed hashes, symlinks, junctions, and paths outside the artifact root cause rejection. Older checkpoints without a dataset fingerprint remain available for inspection. Start a fresh run to obtain weights with verified split provenance.

## Publication and retention

Training prepares a checkpoint in a private directory. A SQLite journal records publication and retirement intents. Recovery replays these entries under a writer lock, so a stopped operation can finish on retry. Each normal save retains the latest `keep_last_n` checkpoints plus any older manual checkpoints. Retirement checks ownership and file membership before removing the active registration. Cleanup unlinks verified checkpoint members individually.

A sudden stop before a publication intent commits can leave a hidden working directory. It has no active registration and loaders ignore it. Preserve private ownership of the artifact root and SQLite database. Operators should inspect abandoned working directories before cleanup.

## Resume

Use `tuneforge resume RUN_ID --root ROOT` with the directory containing `tuneforge.sqlite3`. Stop the previous worker before recovery. The CLI verifies the registered dataset, replays pending checkpoint operations, and loads the latest registered checkpoint. Recovery-critical run inputs are saved before the first update. See [training](training.md) for total-step budgeting and finalization.
