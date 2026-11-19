# Training

A run starts from a registered dataset, a model manifest, and a hashed training plan. The CPU backend constructs a tiny GPT-2 model locally. Full training updates all trainable parameters. PEFT LoRA updates the configured trainable subset. Each update includes a forward pass, finite-loss and gradient checks, gradient clipping, an AdamW step, and measured metrics. QLoRA execution remains unsupported.

## Records and step budget

The optimizer consumes only the IDs in the persisted training split, in manifest order. It cycles through that split when more than one epoch is requested. The local backend uses FP32, one example per optimizer update, one accumulation step and at most 32 tokens. Planning rejects unsupported batch or accumulation values. Set both to one when replanning a workflow that previously specified larger values.

The total update budget is the smaller of `max_steps` and `train_count × epochs`. A resumed run inherits the absolute step number. With a budget of four and a durable checkpoint at step one, the child records steps two, three, and four.

## Recovery

Before the first update, TuneForge persists the dataset and model manifests, training configuration, and plan. A checkpoint identifies the owning run, dataset, model, and configuration. Resume verifies these inputs, creates a linked child run, reloads weights, and initializes a fresh optimizer. The child records only its own updates. Optimizer moments and random-generator state from the parent are not restored.

SQLite is authoritative for run status. Completion follows durable metric and final run exports. A failed final write leaves the parent incomplete or failed. If the final checkpoint already exhausted the budget, recovery can finalize a child from those weights without another update. A completed, exhausted run rejects a further resume. A hard-stopped parent may retain its last recorded state. The child records the recovery lineage. See [checkpoints](checkpoints.md) and [API cancellation](api.md).
