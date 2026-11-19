"""RAAI-SGC grounded Qwen inference on limited VRAM (e.g. RTX 3060 6 GiB).

Keep the interface used by app/gradio_app.py and tests/test_grounding.py.
No model weights are bundled or downloaded until a supported question is asked.
"""
from functools import lru_cache
from pathlib import Path
import json
import re

from src.grounding import UNKNOWN, has_evidence, evidence_messages, cited_sources


@lru_cache(maxsize=1)
def load_generator(model_name):
    """Load a cached 4-bit base model, optionally with a LoRA adapter.

    Intentionally avoids ``transformers.pipeline`` (which imports optional
    vision dependencies in some Transformers installations).
    """
    if not model_name or not str(model_name).strip():
        raise RuntimeError('Enter a Hugging Face model ID or local adapter directory.')
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            'LLM dependencies could not be imported. Launch with '
            '.venv-llm/bin/python and check imports with: '
            'python -c "import torch, transformers, bitsandbytes". '
            f'Original error: {exc}'
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not available in the Python process running Gradio.')

    name = str(model_name).strip()
    adapter = Path(name).expanduser()
    base_id = name
    if adapter.is_dir() and (adapter / 'adapter_config.json').is_file():
        cfg = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
        base_id = cfg.get('base_model_name_or_path')
        if not base_id:
            raise RuntimeError('LoRA adapter_config.json has no base_model_name_or_path.')

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=quant,
        torch_dtype=torch.float16,
        device_map={'': 0},
        low_cpu_mem_usage=True,
    )
    if adapter.is_dir() and (adapter / 'adapter_config.json').is_file():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError('PEFT is needed to load the selected adapter.') from exc
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)

    model.eval()

    def produce(messages, max_new_tokens=220, do_sample=False, return_full_text=False):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        ).to('cuda')
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=min(int(max_new_tokens), 256),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            generated[0][encoded['input_ids'].shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        return [{'generated_text': [{'role': 'assistant', 'content': completion}]}]

    return produce


def extract_generated(response):
    text = response[0].get('generated_text', '')
    if isinstance(text, list):
        return str(text[-1].get('content', '')).strip() if text else ''
    return str(text).strip()


def generate(question, results, model_name, max_new_tokens=220):
    if not has_evidence(results):
        return {'answer': UNKNOWN, 'citations': [], 'mode': 'llm'}
    generator = load_generator(model_name)
    messages = evidence_messages(question, results)
    response = generator(messages, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
    answer = extract_generated(response) or UNKNOWN
    return {'answer': answer, 'citations': cited_sources(answer, results), 'mode': 'llm'}
