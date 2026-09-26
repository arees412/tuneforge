# Training

Every execution requires validated dataset and model manifests plus a hashed `TrainingPlan`. The plan exposes effective batch size, steps, runtime, checkpoint policy, evaluation policy, warnings, and conservative resource estimates.

Supported execution methods:

- `full_fine_tune`: all parameters of the local tiny model are trainable.
- `lora`: documented PEFT APIs inject adapters into validated target modules.
- `qlora`: configuration and capability validation only in routine CPU environments. Execution fails closed unless real CUDA and bitsandbytes capabilities are present; the v0.1 CPU backend still refuses to impersonate quantized training.

The CI model is created with `GPT2Config`, not loaded from a repository. Training performs a real forward pass, finite-loss check, backward pass, finite-gradient check, gradient clipping, AdamW optimizer step, measured metric write, safetensors checkpoint, and checkpoint reload.

Run states are `created → validated → planned → queued → running → evaluating → completed`. `failed` and `cancelled` are terminal branches. Invalid transitions are rejected and persisted. One local training execution may hold the worker lock at a time; TuneForge does not claim distributed scheduling.

Resume validates checkpoint hashes, model-manifest compatibility, and training-configuration compatibility. Weight/adapter state resumes into a new governed run linked to the source run. Optimizer-state resume is intentionally unsupported because v0.1 does not load arbitrary pickle checkpoints.

Precision requests fail before training when the runtime cannot support them. Early stopping, multi-GPU orchestration, model merging, and arbitrary pretrained model loading are intentionally unsupported in v0.1.

