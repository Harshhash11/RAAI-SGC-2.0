# One-folder Google Drive/Colab procedure

1. Upload `RAAI_SGC_2_COMPLETE_PROJECT` directly under **My Drive**.
2. In Colab, open `notebooks/06_colab_drive_inference_and_training.ipynb` from that folder.
3. Choose a T4 GPU under Runtime > Change runtime type, then run notebook cells top-to-bottom.
4. Authorize the explicit Drive mount in notebook. **Do not** mount Drive in notebooks you do not trust.
5. The notebook copies the project to `/content` to avoid slow Drive filesystem access.
6. Run baseline hybrid retrieval, then pretrained Qwen inference. Model downloads
   require network access and GPU allowance; neither has been executed in this package.
7. Only if reviewed and approved training JSONL exists, enable the optional QLoRA cell
   using the exact training command. Otherwise STOP before fine-tuning.
8. Save generated model/checkpoint and experiment results back to the same Drive folder.

Default T4 target is `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit; actual available
VRAM and dependency APIs vary. QLoRA checkpoint is an **adapter**, not complete base weights.
For Ubuntu inference on CPU, the base model may be too slow; default extractive mode stays safe.
