# Training

A run starts from a registered dataset, a model manifest, and a hashed training plan. The CPU backend constructs a tiny GPT-2 model locally. Full training updates all trainable parameters. PEFT LoRA updates the configured trainable subset. Each update includes a forward pass, finite-loss and gradient checks, gradient clipping, an AdamW step, and measured metrics. The v0.1 backend supports text training on CPU. GPU QLoRA execution and full VLM training remain unsupported.

## Records and step budget

The optimizer consumes only the IDs in the persisted training split, in manifest order. It cycles through that split when more than one epoch is requested. The local backend uses FP32, one example per optimizer update, one accumulation step and at most 32 tokens. Planning rejects unsupported batch or accumulation values. Set both to one when replanning a workflow that previously specified larger values.

The total update budget is the smaller of `max_steps` and `train_count × epochs`. A resumed run inherits the absolute step number. With a budget of four and a durable checkpoint at step one, the child records steps two, three, and four.

## Executable configuration

TuneForge v0.1 uses a constant learning rate with zero warmup. Linear scheduling, epoch-based saving, automatic evaluation, early stopping and sparse metric logging are unsupported. Planning and training reject requests for these behaviors. A new workflow defaults to `evaluation_strategy="none"`. Use `tuneforge evaluate RUN_ID --root ROOT` for explicit held-out scoring.

| Field | Accepted for CPU execution | Actual behavior | Support | Test evidence |
|---|---|---|---|---|
| `epochs` | Integer 1–100 | Bounds the number of passes over the training IDs. | Supported | Epoch budget |
| `max_steps` | Integer 1–100,000 | Caps the absolute update number. Resume continues from the last durable checkpoint. | Supported | Four updates, recovery |
| `learning_rate` | Greater than 0, at most 1 | Sets the AdamW rate on every update, including after restart. | Supported | Four updates, recovery |
| `weight_decay` | 0–1 inclusive | Sets weight decay on the AdamW parameter group. | Supported | Four updates |
| `warmup_ratio` | 0 | Uses the configured rate from the first update. | Zero only | Rejection, four updates |
| `per_device_batch_size` | 1 | Passes one training record to each forward pass. | One only | Four updates, split trace |
| `gradient_accumulation` | 1 | Clears gradients and performs one optimizer update for each record. | One only | Four updates, split trace |
| `max_sequence_length` | Integer 4–32 | Caps the byte-tokenized model input length. | Supported | Four updates, direct rejection |
| `optimizer` | `adamw` | Uses PyTorch AdamW on trainable parameters. | AdamW only | Four updates, rejection |
| `scheduler` | `constant` | Keeps the configured rate for the entire run. | Constant only | Four updates, rejection |
| `precision` | `fp32` | Requires PyTorch’s FP32 default dtype and disabled CPU autocast. | FP32 only | Four updates, precision rejection |
| `gradient_clipping` | Finite number greater than 0 | Clips the trainable gradients by global norm before each update. | Supported | Four updates, overflow rejection |
| `seed` | Integer from −2⁶³ through 2⁶⁴−1 | Seeds Python and PyTorch for execution and initializes the model. Dataset splitting has its own seed. | Supported | Four updates, rejection |
| `logging_steps` | 1 | Persists a training metric after each successful update. | One only | Four updates, rejection |
| `evaluation_strategy` | `none` (default) | Runs no scheduled scoring. Explicit CLI evaluation uses registered test IDs. | Explicit evaluation only | Four updates, split trace |
| `save_strategy` | `steps` | Uses absolute step numbers for checkpoint saving. | Steps only | Four updates, rejection |
| `checkpoint_frequency` | Positive integer | Saves at each multiple and at the final budget step. | Supported | Four updates, recovery |
| `early_stopping_patience` | `null` or omitted | Runs to the step budget unless interrupted, cancelled, or failed. | Early stopping unsupported | Rejection, epoch budget |
| `keep_last_n` | Integer 1–100 | Keeps the latest N checkpoints for the run and protects older manual checkpoints. | Supported | Retention, recovery |

“Four updates” refers to `test_four_real_updates_match_configuration`, which records the optimizer group, clipping, input shapes, metrics and checkpoints in default and explicitly configured API runs. The same [configuration tests](../tests/test_config_contract.py) cover the epoch budget, public and direct rejection, overflow and persisted settings. The [execution tests](../tests/test_governed_execution.py) trace splits and recover after process exit. The [checkpoint tests](../tests/test_checkpoint_recovery.py) exercise retention and manual protection.

## Recovery

Before the first update, TuneForge persists the dataset and model manifests, training configuration, and plan. A checkpoint identifies the owning run, dataset, model, and configuration. Resume verifies these inputs, creates a linked child run, reloads weights, and initializes a fresh optimizer. The child records only its own updates. Optimizer moments and random-generator state from the parent are not restored.

SQLite is authoritative for run status. Completion follows durable metric and final run exports. A failed final write leaves the parent incomplete or failed. If the final checkpoint already exhausted the budget, recovery can finalize a child from those weights without another update. A completed, exhausted run rejects a further resume. A hard-stopped parent may retain its last recorded state. The child records the recovery lineage. See [checkpoints](checkpoints.md) and [API cancellation](api.md).

Resume validates the original saved settings before replaying checkpoint operations. A run that recorded an unsupported setting, including the previous default `evaluation_strategy="steps"`, cannot resume under v0.1. Start a new run with supported settings to train further. The explicit evaluation command can still load a historical checkpoint with its original configuration digest and verified dataset. It leaves the recorded settings intact.
