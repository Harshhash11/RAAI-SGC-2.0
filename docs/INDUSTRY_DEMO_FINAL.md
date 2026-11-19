# RAAI-SGC 2.0 — Industry Capstone Demo Handoff

This is a **patch for the existing Ubuntu project**, not a new project. Apply at the project root. It does not replace training data, model weights, evaluation outputs, or user review decisions.

## Launch

```bash
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY
./run_industry_demo.sh
```

Open the local URL printed in the terminal (typically http://127.0.0.1:7860).

## Demonstration flow

1. Ask `What is Kubernetes?` with `Extractive (offline)` and `hybrid` / `tfidf`.
2. Switch to `Local LLM` and `Qwen/Qwen2.5-1.5B-Instruct`, ask the same question.
3. Choose `Fine-tuned Qwen (RAAI-SGC)` to use `models/checkpoints/qlora_final_44` and ask again.
4. Show the interactive graph tab and retrieval diagnostics. Demonstrate that graph expansion may activate, but does not always change top-3 results.
5. Show retrieved source links and distinguish them from **explicit answer citations**. If the generated answer omits `[1]` markers, its source links are labeled *retrieved*, not claimed as citations.
6. Ask a question unsupported by the dataset; describe the evidence threshold and its limitations.

Stop any other GPU-heavy applications while running a 5.8 GB VRAM GPU.

## Pre-demo checks

```bash
.venv-llm/bin/python industry_check.py
.venv-llm/bin/python -m unittest discover -s tests -p 'test_industry_handoff.py' -v
```

The offline checker validates file layout, adapter presence, train file, and UI selection without running the model. Complete one actual LLM answer in Gradio to validate runtime CUDA inference.

## Known limitations (do not hide these in a demo)

- The dataset has 145 source documents and 290 draft QA pairs; only 44 reviewed training examples were exported in the user's last run.
- QLoRA adapter was trained using 44 examples. That establishes pipeline functionality, not superiority over pretrained Qwen.
- The initial held-out 5-question comparison yielded graph activation on 2 queries and changed top-3 retrieval on 1. No verified relevance or answer-quality improvement has been established.
- LLM outputs sometimes lack explicit `[1]` source markers. Retrieved document links are now displayed independently; **retrieval does not prove a claim**.
- The existing app is a **local single-user prototype**, not a production service. It does not have authentication, rate limits, monitoring, multiuser resource management, or an independently validated benchmark.
- Source licensing and attribution reviews depend on user-entered review decisions and are not cryptographically independently verified by the checker.

## Emergency rollback

The install commands create backups before replacing `app/gradio_app.py` and `src/llm.py` is not replaced by this patch. To roll back, restore `app/gradio_app.py` from the timestamped backup file.
