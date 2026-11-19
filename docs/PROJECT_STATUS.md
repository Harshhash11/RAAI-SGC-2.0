# Status ledger (truthful state)

| Component | State | Evidence or missing step |
|---|---|---|
| Dataset 145/4771/290 | Bundled | Real Part 1 final master file |
| Dataset structural validation | Previously executed in Part 1 | Recheck in unified package |
| TF-IDF retrieval | Implemented and locally tested | Source citations and scores |
| Weighted and unweighted graph retrieval | Implemented | Local tests, but no reviewed comparative benchmark |
| Semantic embeddings | Optional implementation | Must download model and run locally |
| Offline extractive Gradio chat | Implemented | Install Gradio and run locally |
| Generative LLM chat | Optional implemented | Requires a chosen compatible model / hardware |
| QLoRA | Experimental code | Not run; zero approved Q&A; needs CUDA |
| Gold evaluation | Infrastructure only | Zero human-reviewed gold questions |
| Research paper | Planning and source guidelines | Do not fabricate results; user-authored manuscript required |
| Capstone ready for live assessment | Partial | Test deployed Gradio and model before demonstration |

- Project-specific EDA and retrieval-experiment notebooks are runnable without gold labels; benchmark numbers remain gated.
