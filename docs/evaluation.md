# Evaluation

`EvaluationSuite` is versioned and contains deterministic cases with exact-match, JSON-format, or numeric-loss rules. Case and aggregate results are persisted. Baseline/candidate comparison requires the same suite ID and version, computes metric deltas, and applies explicit regression tolerance.

The end-to-end demo measures language-model loss for a locally constructed random baseline and a trained checkpoint on every record in the registered test split, reporting their mean loss. Those values prove execution and comparison mechanics only. They are not general benchmarks and must not be used to claim production model quality.

`assert_no_evaluation_contamination` rejects exact normalized prompt/target overlaps between training and evaluation records. Dataset split leakage is also checked before the plan can execute.

CLI model evaluation verifies the current source fingerprint and checkpoint ownership before scoring the persisted test split. An empty test split is rejected. The HTTP prediction-scoring endpoint scores the cases and predictions supplied in its request.

Registry approval requires immutable evaluation evidence whose subject matches the candidate ID or its owning run. Changes of registry status and their evidence references are audited together. A passing score is a separate policy decision. TuneForge never chooses a winner without an explicit comparison rule, and it records warnings when comparable metrics are absent.
