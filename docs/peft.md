# PEFT and LoRA

TuneForge uses Hugging Face PEFT through its documented `LoraConfig`, `get_peft_model`, `save_pretrained`, and `PeftModel.from_pretrained` APIs. PEFT and LoRA are upstream technologies; TuneForge does not claim them as original inventions.

The TuneForge LoRA manifest records rank, alpha, dropout, target modules, bias policy, task type, and modules to save. Rank and dropout are strictly validated. Target names are checked against actual linear/Conv1D modules before training.

For every adapter execution TuneForge computes total, trainable, and trainable-percentage values from the actual model. CI verifies that adapter injection produces a positive trainable count smaller than the total, that gradients and optimizer steps occur, that safetensors adapter files are written, and that the adapter reloads over the same deterministic base configuration.

QLoRA configuration captures 4-bit loading, NF4/FP4, compute dtype, and double quantization. TuneForge does not substitute a CPU simulation for bitsandbytes/CUDA execution. GPU QLoRA remains optional and unverified in routine CI.

