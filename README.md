# RAAI-SGC 2.0

**Retrieval-Augmented AI with a Semantic Graph for IT Knowledge Assistance**  
*A weighted Markov–TF-IDF graph-augmented question-answering prototype with local Qwen inference and QLoRA fine-tuning.*

RAAI-SGC 2.0 is a local, single-user capstone application for asking technical questions across **Artificial Intelligence / Machine Learning, Cybersecurity, Cloud / DevOps, and Software Engineering**. It combines lexical document retrieval, concept-graph expansion, evidence-grounded generation, an interactive graph display, and a fine-tuned language-model adapter in a Gradio interface.

> **Project status:** Functional local industry-capstone prototype. Pretrained and fine-tuned model inference, retrieval, graph visualization, and the QLoRA training pipeline have been demonstrated on an NVIDIA RTX 3060 Laptop GPU. This is **not** a production-hosted enterprise service or a clinically/security-certified decision system. A larger independent benchmark is future work.

## Features

- **Three retrieval methods:** `vector` (TF-IDF), `unweighted` graph retrieval, and `hybrid` weighted graph-assisted retrieval.
- **Retrieval backends:** TF-IDF and an optional semantic backend (availability depends on installed dependencies).
- **Response modes:** offline/extractive answers and local Qwen-generated answers.
- **Generator selection:** pretrained `Qwen/Qwen2.5-1.5B-Instruct`, fine-tuned LoRA adapter, or a custom Hugging Face model/adapter path (depending on the installed UI revision).
- **4-bit inference:** NF4 quantization via Transformers and bitsandbytes for low-VRAM NVIDIA GPUs.
- **Interactive concept graph:** retrieved source pages, extracted concepts, weighted edges, and retrieval diagnostics.
- **Transparent sources:** retrieved-source links are displayed as retrieval provenance. A retrieved source is **not** automatically proof of every sentence in a generated answer.
- **Review-gated QLoRA:** source and Q&A review checks must pass before training examples are exported.
- **Repeatable experiments:** approved SFT export, QLoRA training, and matched-question configuration comparisons.

## Architecture

```text
Wikipedia-derived technical source documents
                │
                ▼
     Preprocessing / master.csv
      ├── SOURCE records
      ├── weighted graph EDGE records
      └── QA_DRAFT records → review queues → approved SFT examples
                │
                ▼
         HybridRetriever
      ├── TF-IDF vector search
      ├── unweighted graph mode
      └── weighted Markov–TF-IDF graph expansion
                │
        Top-k retrieved evidence
           ┌────┴─────┐
           ▼          ▼
    Extractive mode   Local Qwen (4-bit)
                      ├── pretrained model
                      └── optional QLoRA adapter
           └────┬─────┘
                ▼
        Gradio answer + diagnostics
          + source links + graph
```

The project uses a graph-based edge-weighting approach described as:

\[
W(A,B) = P(A \to B)\left(1 + \operatorname{TFIDF}(B)\right)
\]

Here, the Markov-style transition term and TF-IDF term help prioritize concept associations. The resulting graph is built from extracted source-to-concept associations; these associations should **not** be interpreted as independently verified semantic facts.

## Data and demonstrated scope

| Item | Recorded project scope |
| --- | ---: |
| Source documents | 145 |
| Weighted graph edge records | 4,771 |
| Draft Q&A records | 290 |
| Approved examples used for the demonstrated QLoRA run | 44 |
| Initial matched-question evaluation set | 5 questions |
| Domain coverage | AI/ML, Cybersecurity, Cloud/DevOps, Software Engineering |

The underlying `data/processed/master.csv` contains different record types. Draft Q&A rows are **not** approved training labels merely because they appear in the dataset.

The training experiment used 44 records that passed the project's source and Q&A approval-field checks. Those checks represent recorded review decisions, not an external audit of truth, licensing, or overall answer quality.

## Requirements

- **Operating system:** Ubuntu Linux (primary tested development environment).
- **Python:** 3.12 for the demonstrated local setup.
- **GPU for local Qwen:** NVIDIA CUDA GPU. The demonstrated system used an **RTX 3060 Laptop GPU with 5.8 GiB available VRAM**.
- **CPU-only operation:** Extractive retrieval and graph exploration can run without local GPU generation.
- **Hugging Face access:** Internet connection may be required the first time model files are downloaded.

Two Python virtual environments were used during development:

- `.venv` for general app/data and CPU development tasks.
- `.venv-llm` for CUDA PyTorch, Transformers, bitsandbytes, PEFT, datasets, and GPU Gradio operation.

Known working GPU-stack versions during development included PyTorch `2.5.1+cu121`, Transformers `4.49.0`, Accelerate `1.5.2`, bitsandbytes `0.45.5`, PEFT `0.15.2`, and Gradio `5.49.1`. Your actual lockfile/requirements, if present, should take precedence; versions can change.

## Quick start (existing prepared Ubuntu project)

```bash
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY
source .venv-llm/bin/activate
python -m app.gradio_app
```

If your final handoff includes the executable launcher, you may instead run:

```bash
chmod +x run_industry_demo.sh
./run_industry_demo.sh
```

Open the local URL shown in the terminal (typically **http://127.0.0.1:7860**).

### Fresh clone setup

Python virtual environments, downloaded base model weights, and local trained adapters are intentionally **not automatically published to GitHub**. After cloning:

1. Install Python 3.12, Git, a compatible NVIDIA driver/CUDA runtime (for local generation), and your package prerequisites.
2. Create and activate a virtual environment: `python3 -m venv .venv-llm && source .venv-llm/bin/activate`.
3. Install the project dependencies from the repository's pinned requirements/configuration **if present**. The GPU-enabled environment must contain compatible versions of `torch`, `transformers`, `accelerate`, `bitsandbytes`, `peft`, `datasets`, `gradio`, and the retriever/graph libraries used in `src/`.
4. Make `data/processed/master.csv` available from the appropriately licensed project data distribution or regenerate it using the supplied dataset-engineering pipeline.
5. For fine-tuned inference, place your locally trained adapter under `models/checkpoints/qlora_final_44` or point the custom-adapter field at another compatible local adapter directory.
6. Launch with `python -m app.gradio_app`.

**Note:** A GitHub clone alone is not guaranteed to run GPU generation until the model dependencies, data, and adapter are provisioned. Do not commit `.venv*` directories or Hugging Face cache/model weights.

## Using the application

1. Enter a technical question, such as **“What is Kubernetes?”**.
2. Choose `vector`, `unweighted`, or `hybrid` retrieval.
3. Select `tfidf` or the optional semantic backend.
4. Choose `Extractive (offline)` or `Local LLM`.
5. In local LLM mode, choose pretrained Qwen or your custom/fine-tuned adapter. The demonstrated adapter directory is:

   ```text
   models/checkpoints/qlora_final_44
   ```

6. Select **Ask + update graph** to generate an answer and update the evidence graph, or **Update graph only (no GPU)** to explore retrieved associations without LLM generation.

The diagnostics section reports retrieved source titles, vector scores, graph scores, and whether graph expansion activated.

## Repository layout

```text
RAAI_SGC_2_COMPLETE_PROJECT_READY/
├── app/
│   └── gradio_app.py              # Web interface and retrieval diagnostics
├── src/
│   ├── llm.py                     # Quantized local Qwen / optional PEFT adapter
│   ├── hybrid.py                  # HybridRetriever and retrieval data loading
│   ├── grounding.py               # Evidence prompt construction and citations
│   ├── graph_viz.py               # Interactive graph visualization
│   └── answers.py                 # Extractive answer mode
├── data/
│   ├── processed/master.csv       # Canonical mixed source/edge/QA dataset
│   ├── review/                    # Human review queues (keep private by default)
│   ├── training/                  # Approved SFT export (keep private by default)
│   └── evaluation/                # Prediction outputs / review fields
├── dataset_engineering/          # Review, validation, data export utilities
├── training/
│   ├── prepare_sft.py             # Prepare SFT examples from approved QA
│   ├── validation.py              # Training provenance checks
│   └── train_qlora_colab.py       # 4-bit PEFT/QLoRA trainer (also runs locally)
├── evaluation/
│   └── compare_models.py          # Matched-question configuration evaluation
├── tests/                         # Unit tests
├── models/checkpoints/           # LOCAL generated adapters (ignored by Git)
├── README.md
└── .gitignore
```

Other notebooks, reports, launchers, and dataset-engineering files may also be included depending on the final project snapshot.

## Human review and QLoRA fine-tuning

The canonical export requires approved values in both source and question review queues:

- **Source checks:** `domain_relevance`, `factual_accuracy`, `text_completeness`, `license_review`, `attribution_review`, plus a nonempty reviewer.
- **Q&A checks:** `answer_factuality`, `answer_relevance`, `answer_completeness`, `evidence_adequacy`, plus nonempty `approved_answer` and reviewer.

An AI-generated suggestion is not automatically an approved answer. Recording `APPROVED` is a human review decision and must not be fabricated or inferred from a formatter or token-overlap heuristic.

From an appropriately reviewed local dataset:

```bash
source .venv/bin/activate
python dataset_engineering/export_approved.py
python -m training.prepare_sft
```

Then switch to a CUDA-enabled environment:

```bash
source .venv-llm/bin/activate
python -m training.train_qlora_colab \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/training/sft_train.jsonl \
  --output models/checkpoints/qlora_final_44 \
  --epochs 1 \
  --max-steps 11 \
  --max-length 256
```

The example above reproduces the demonstrated **44-example, 11-step** experiment configuration; it is not a claim of a universal optimum. The QLoRA adapter is separate from the base-model weights.

## Evaluation and results

The project includes an evaluation runner for three generation configurations:

| Name | Retrieval | Generator |
| --- | --- | --- |
| `baseline` | TF-IDF vector | Pretrained Qwen |
| `graph` | Weighted hybrid | Pretrained Qwen |
| `finetuned` | Weighted hybrid | QLoRA-adapted Qwen |

```bash
source .venv-llm/bin/activate
python -m evaluation.compare_models --config baseline
python -m evaluation.compare_models --config graph
python -m evaluation.compare_models --config finetuned
```

The evaluation runner protects existing output files from overwrite. Move or rename previous `data/evaluation/predictions_*.csv` files before repeating runs.

### Preliminary five-question run

| Metric | Baseline | GraphRAG | Fine-tuned GraphRAG |
| --- | ---: | ---: | ---: |
| Questions evaluated | 5 | 5 | 5 |
| Mean response latency (seconds) | 2.873 | 2.815 | 2.333 |
| Graph expansion activated | 0 | 2 | 2 |
| Answers with recorded citations | 0 | 0 | 0 |

Graph-assisted retrieval altered the top-three retrieved source order/content for **1 of 5 questions**, while graph activation occurred for **2 of 5**. The fine-tuned model showed a lower average response latency in this small run, but the sample is too small to claim a reliable performance advantage. Model-loading time was excluded from the per-question measurements.

The demonstrated fine-tuning run produced a **training loss of approximately 1.7528**; this is not a test-accuracy metric. Independent human correctness, relevance, and evidence-support ratings have not been established for the reported five-question evaluation. The evaluated source-linked generation path recorded no explicit citations; newer UI variants can show retrieved source links, which should not be confused with claim-level citations.

## Testing

Run the project's unit tests:

```bash
source .venv-llm/bin/activate
python -m unittest discover -s tests -p 'test_*.py' -v
```

If present in the final handoff, the offline deployment check is:

```bash
python industry_check.py
```

For the live demo, verify model selection, fine-tuned adapter loading, one supported question, an unsupported question, source-link display, and interactive graph rendering.

## Troubleshooting

| Symptom | Suggested action |
| --- | --- |
| `No module named 'training'` | Launch from repository root with `python -m training.train_qlora_colab`, not `python training/train_qlora_colab.py`. |
| `No module named 'datasets'` | In `.venv-llm`, install a compatible `datasets` version (the demonstrated fix was `datasets==3.6.0`). |
| `CUDA available: False` | Check the active Python environment, driver, CUDA-enabled PyTorch, and `nvidia-smi`. |
| CUDA out of memory | Close other GPU programs, reduce maximum sequence length, and check 4-bit quantization. |
| Fine-tuned adapter not found | Verify the directory contains `adapter_config.json` and `adapter_model.safetensors`. |
| No approved training examples | Check BOTH `QA_REVIEW_QUEUE.csv` and `SOURCE_REVIEW_QUEUE.csv`; automated suggestions are not approvals. |
| Existing evaluation CSV blocks rerun | Archive or rename previous prediction CSVs; do not overwrite reviewed results. |
| Generated answer lacks citations | Show retrieved-source provenance separately and verify claims against source text before treating answers as grounded. |

## Responsible use, licensing, and release

- Source material is Wikipedia-derived; check each source's applicable license, attribution, and redistribution obligations before releasing the processed dataset.
- Human review queues may include reviewer names, decisions, and dates. They are excluded from the recommended public repository by default.
- Local model weights, adapters, checkpoints, Hugging Face caches, and environment directories are excluded from the recommended public repository by default.
- The app provides technical assistance; generated answers may be wrong, incomplete, or unsupported. The graph represents extracted associations, not validated causal knowledge.
- **Recommended initial GitHub visibility: private** until the included data, files, and attribution are audited.
- Do not publish secrets, tokens, private credentials, or personal review records.

## Future work

- A larger, independently adjudicated held-out benchmark.
- Comparative relevance evaluation of TF-IDF, unweighted graph, and weighted graph retrieval (e.g., Recall@k, MRR, nDCG with genuine relevance labels).
- Better claim-to-evidence citation grounding and answer abstention.
- Multi-user deployment, access control, observability, and reproducible dependency locking.
- Research paper and presentation, maintained separately from the finished local industry demo.

## Acknowledgments

- [Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for the base language model.
- [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [Datasets](https://github.com/huggingface/datasets).
- [Gradio](https://www.gradio.app/) for the local web interface.
- Wikipedia contributors for the original encyclopedic source material, subject to source-specific attribution requirements.

---

**RAAI-SGC 2.0** — Local graph-augmented IT knowledge assistant and capstone prototype.
