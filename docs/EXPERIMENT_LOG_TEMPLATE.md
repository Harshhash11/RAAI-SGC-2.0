# Experimental log template (fill with real execution evidence)

| Field | Fill with observed value |
|---|---|
| Date (UTC) | |
| Git commit / project SHA256 | |
| Source corpus SHA256 | See `reports/generated/final_readiness_audit.json` |
| Base model/revision | |
| GPU/device and VRAM | |
| Python, torch, transformers, peft versions | |
| Training examples and review evidence | |
| Train/validation/test protocol | |
| Seed, epochs, steps, batch, learning rate | |
| Adapter path and size | |
| Train and validation loss | |
| Gold QA sample size | |
| Vector-only Recall@1/@5 and MRR | |
| Unweighted graph Recall@1/@5 and MRR | |
| Weighted graph Recall@1/@5 and MRR | |
| Human-rated faithfulness/correctness | |
| Runtime and memory | |
| Key limitations and failure cases | |

**Do not insert guessed values.** A successful fine-tuning run is not evidence of improved answering without held-out evaluation.
