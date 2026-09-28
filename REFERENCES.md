# Research references and implementation boundary

Reviewed on 2026-09-26. These primary sources informed architecture and public API use; they are not source-code inputs to this repository.

- [LLaMA-Factory repository and documentation](https://github.com/hiyouga/LLaMA-Factory): architecture and supported-workflow research only.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index): adapter API and checkpoint boundary.
- [PEFT LoRA configuration reference](https://huggingface.co/docs/peft/package_reference/lora): public LoRA configuration parameters.
- [Transformers documentation](https://huggingface.co/docs/transformers/index): configuration-based local model construction and model interfaces.
- [Transformers bitsandbytes quantization guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes): QLoRA capability requirements and `BitsAndBytesConfig` research.
- [Accelerate documentation](https://huggingface.co/docs/accelerate/index): runtime and future backend research.
- [TRL documentation](https://huggingface.co/docs/trl/index): optional trainer-interface research; TRL is not required by routine execution.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html): deterministic control limits.
- [safetensors documentation](https://huggingface.co/docs/safetensors/index): safe tensor serialization.
- [FastAPI documentation](https://fastapi.tiangolo.com/): typed local HTTP service.
- [Typer documentation](https://typer.tiangolo.com/): command-line interface.

## Independent-implementation boundary

TuneForge was designed and written from the specification and documented public APIs. It does not copy source code, tests, prompts, README text, model cards, branding, benchmark claims, or commit history from LLaMA-Factory, PEFT, TRL, Transformers, Accelerate, PyTorch, or other upstream projects. Their names identify third-party dependencies or research sources, not TuneForge authorship.

This project does not redistribute upstream model weights or datasets. Operators are responsible for validating separate model and dataset license terms.
