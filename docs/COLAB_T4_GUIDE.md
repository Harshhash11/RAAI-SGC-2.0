# Colab T4 integration — existing unified folder

1. On Ubuntu keep `RAAI_SGC_2_COMPLETE_PROJECT` as your only project folder.
2. This add-on introduces `notebooks/05_colab_t4_llm_qlora.ipynb` and `training/train_qlora_colab.py`.
3. ZIP your existing folder: from `~/Downloads`, run `zip -r -q RAAI_SGC_2_COMPLETE_PROJECT.zip RAAI_SGC_2_COMPLETE_PROJECT -x '*/.venv/*' '*/__pycache__/*' '*/.pytest_cache/*'`.
4. Visit https://colab.research.google.com and upload `notebooks/05_colab_t4_llm_qlora.ipynb` via File > Upload notebook.
5. Select Runtime > Change runtime type > T4 GPU; run cells in order. The notebook asks for the entire unified ZIP.
6. Execute the pretrained model and evidence-grounded query first. QLoRA training is blocked until your reviewed training exporter yields 20+ genuinely approved records.
7. After training, download the adapter ZIP and store it in `models/checkpoints/qlora_colab/`. It requires the same base model for inference. Your Ubuntu machine may need GPU/RAM; otherwise run inference on Colab.

**Do not upload credentials or private data in public Colab notebooks.** The 145-source master CSV includes unreviewed material; cite source URLs but do not call its answers verified.

The notebook and training script have been generated and syntax/schema-checked locally, not executed on a Colab T4 in this environment.
