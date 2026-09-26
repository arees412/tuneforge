# Evaluation

`EvaluationSuite` is versioned and contains deterministic cases with exact-match, JSON-format, or numeric-loss rules. Case and aggregate results are persisted. Baseline/candidate comparison requires the same suite ID and version, computes metric deltas, and applies explicit regression tolerance.

The end-to-end demo measures language-model loss for a locally constructed random baseline and a trained checkpoint on a held-out fixture record. Those values prove execution and comparison mechanics only. They are not general benchmarks and must not be used to claim production model quality.

`assert_no_evaluation_contamination` rejects exact normalized prompt/target overlaps between training and evaluation records. Dataset split leakage is also checked before the plan can execute.

Registry approval requires an existing evaluation record. TuneForge never chooses a winner without an explicit comparison rule, and it records warnings when comparable metrics are absent.

