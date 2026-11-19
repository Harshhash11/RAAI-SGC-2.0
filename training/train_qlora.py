"""Optional NVIDIA CUDA QLoRA experiment. Never trains on drafts.
Requires carefully reviewed approved training examples and installed GPU dependencies.
"""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',default='data/training/sft_train.jsonl');p.add_argument('--output',default='models/checkpoints/qlora');p.add_argument('--epochs',type=int,default=1);p.add_argument('--max-steps',type=int,default=100)
    a=p.parse_args()
    if a.epochs<=0 or a.epochs>25:raise ValueError('epochs must be 1..25')
    rows=[json.loads(line) for line in Path(a.data).read_text(encoding='utf8').splitlines() if line.strip()] if Path(a.data).exists() else []
    if len(rows)<20:raise RuntimeError(f'Only {len(rows)} approved training records; require at least 20 for smoke training, preferably substantially more reviewed data.')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('QLoRA requires CUDA GPU; run on Colab T4 or local NVIDIA GPU.')
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments,Trainer,DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,quantization_config=quant,device_map='auto')
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules='all-linear'))
    def tokenize(row):
        msgs=row['messages']
        if getattr(tokenizer,'chat_template',None):
            content=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
        else:content='\n'.join(f"{m['role'].upper()}: {m['content']}" for m in msgs)
        return tokenizer(content,truncation=True,max_length=512)
    tokenized=Dataset.from_list(rows).map(tokenize,remove_columns=list(rows[0].keys()))
    args=TrainingArguments(output_dir=a.output,per_device_train_batch_size=1,gradient_accumulation_steps=4,num_train_epochs=a.epochs,max_steps=a.max_steps,learning_rate=2e-4,logging_steps=10,save_strategy='epoch',report_to='none',fp16=True)
    trainer=Trainer(model=model,args=args,train_dataset=tokenized,data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False))
    trainer.train();model.save_pretrained(a.output);tokenizer.save_pretrained(a.output)
    print('Adapter saved to',a.output)
if __name__=='__main__':main()
