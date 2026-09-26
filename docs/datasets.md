# Datasets

TuneForge reads bounded local UTF-8 JSON, JSONL, and CSV files. JSON must be an array or an object with a `records` array. JSONL requires one object per non-empty line. CSV headers become record fields.

Supported schemas are strict:

- instruction: `instruction`, optional `input`, `output`, optional `system`
- chat: alternating `user` and `assistant` messages, with an optional first `system` message
- prompt/completion: `prompt`, `completion`
- preference: `prompt`, distinct `chosen` and `rejected`

All schemas normalize to `CanonicalTrainingRecord`. Unsupported fields produce an issue instead of being silently discarded. Raw samples are not written to logs or audit events.

Fingerprint inputs include the source SHA-256, normalization and schema versions, cleaned record count, and split manifest. Exact and normalized-text duplicates are reported with kept/duplicate IDs and a `keep_first` decision. Records are not silently removed: callers receive the duplicate report and the manifest records the decision.

The default stable-hash split uses each canonical record ID plus a seed. A seeded-random strategy is also available. The manifest records ratios, seed, algorithm, IDs, counts, and split digest. Leakage checks detect cross-split identical records, prompts, and targets. A training plan is rejected when leakage remains.

The quality report includes valid/invalid counts, duplicates, empty targets, lengths, violations, leakage, sensitivity warnings, and license-metadata availability. The built-in whitespace tokenizer provides offline deterministic token-count analysis; it is not presented as the production model tokenizer.

Dataset license fields are operator-supplied metadata. `unknown` is the safe default. TuneForge does not infer licensing rights from a source location and does not provide legal advice.

