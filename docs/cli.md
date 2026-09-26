# CLI

`tuneforge --help` lists only implemented commands.

- `dataset validate|inspect|split <path>`
- `plan <workflow.json>` and `train <workflow.json>`
- `resume <run-id>`, `evaluate <run-id>`, and `compare <baseline-eval> <candidate-eval>`
- `checkpoints <run-id>`, `registry list`, and `evidence <run-id>`
- `capabilities`, `demo`, and `eval`

`demo` executes one LoRA workflow. `eval` executes all twelve specification acceptance scenarios, including real full and LoRA training plus negative security/failure paths. Both write only to the selected ignored root.

