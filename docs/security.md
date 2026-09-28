# Security and threat model

TuneForge treats datasets, model repositories, checkpoints, credentials, and artifact paths as untrusted boundaries.

Controls in v0.1 include:

- remote model code denied unless a model spec records explicit operator opt-in;
- no remote model download in CI and no automatic Hugging Face Hub publication;
- no arbitrary pickle checkpoint loading;
- bounded dataset bytes, record count, record text, sequence length, batch size, steps, and retention count;
- canonical path confinement, safe artifact identifiers, no unrelated overwrite, and tested traversal rejection;
- secret-pattern redaction for Bearer values, Hugging Face tokens, API keys, passwords, and token-like fields;
- audit events exclude raw training records and redact details before persistence;
- one active local trainer per worker;
- ignored artifact, dataset, cache, credential, checkpoint, and large-model patterns;
- CI scans tracked files for credential signatures and prohibited large/binary model artifacts.

Residual risks include dependency or model supply-chain compromise, operator-provided malicious content, denial of service within configured limits, sensitive data intentionally placed in metadata, hardware-specific numerical differences, and future publisher misuse. Operators must isolate sensitive datasets, restrict filesystem permissions, verify source and artifact licenses, pin reviewed dependencies, and protect tokens outside TuneForge.

TuneForge does not claim SOC 2, HIPAA, GDPR, PCI DSS, or any other compliance certification. These controls are engineering safeguards, not a compliance determination or legal advice.
