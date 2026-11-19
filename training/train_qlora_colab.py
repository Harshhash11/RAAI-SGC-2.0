"""Colab T4 QLoRA: SFT on assistant tokens only, using reviewed data.

Never trains on unreviewed QA_DRAFT or held-out evaluation records.
"""
import argparse
import json
from pathlib import Path


def load_approved(path):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            row=json.loads(line)
            messages=row.get('messages',[])
            if [m.get('role') for m in messages] != ['system','user','assistant']:
                raise ValueError(f'Line {i}: expected system/user/assistant messages')
            if not all(isinstance(m.get('content'),str) and m['content'].strip() for m in messages):
                raise ValueError(f'Line {i}: empty messages')
            if not row.get('source_id'):
                raise ValueError(f'Line {i}: source_id required')
            rows.append(row)
    if len(rows)<20:
        raise RuntimeError(f'Only {len(rows)} reviewed records. Need 20+ for a GPU smoke test; considerably more for useful training.')
    return rows


def encode_example(row,tokenizer,max_length=512):
    msgs=row['messages']
    prefix=tokenizer.apply_chat_template(msgs[:2],tokenize=False,add_generation_prompt=True)
    answer=msgs[2]['content'].strip()+tokenizer.eos_token
    prefix_ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
    answer_ids=tokenizer(answer,add_special_tokens=False)['input_ids']
    if len(answer_ids)>=max_length:
        raise ValueError('Answer is too long; review and shorten it before training')
    prefix_ids=prefix_ids[-(max_length-len(answer_ids)):]
    ids=prefix_ids+answer_ids
    labels=[-100]*len(prefix_ids)+answer_ids
    assert any(x!=-100 for x in labels)
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--data',default='data/training/sft_train.jsonl')
    p.add_argument('--output',default='models/checkpoints/qlora_colab')
    p.add_argument('--approved-export',default='data/review/APPROVED_QA_ONLY.csv')
    p.add_argument('--master',default='data/processed/master.csv')
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--max-steps',type=int,default=30)
    p.add_argument('--max-length',type=int,default=512)
    a=p.parse_args()
    if not 1<=a.epochs<=25: raise ValueError('epochs must be between 1 and 25')
    if a.max_steps<1: raise ValueError('max-steps must be positive')
    rows=load_approved(a.data)
    from training.validation import validate_training_provenance
    provenance = validate_training_provenance(rows,a.approved_export,a.master)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Requires NVIDIA CUDA GPU (choose Colab T4)')
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,TrainingArguments,Trainer,DataCollatorForSeq2Seq
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from datasets import Dataset
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    data=[encode_example(x,tokenizer,a.max_length) for x in rows]
    dataset=Dataset.from_list(data)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,quantization_config=quant,device_map='auto')
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules='all-linear'))
    args=TrainingArguments(output_dir=a.output,per_device_train_batch_size=1,gradient_accumulation_steps=4,num_train_epochs=a.epochs,max_steps=a.max_steps,learning_rate=1e-4,logging_steps=5,save_strategy='steps',save_steps=25,save_total_limit=2,report_to='none',fp16=True,gradient_checkpointing=True,remove_unused_columns=False)
    collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,label_pad_token_id=-100,padding=True)
    trainer=Trainer(model=model,args=args,train_dataset=dataset,data_collator=collator)
    result=trainer.train()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output/'experiment_metadata.json').write_text(json.dumps({'base_model':a.model,'approved_examples':len(rows),'provenance':provenance,'epochs':a.epochs,'max_steps':a.max_steps,'assistant_only_loss':True,'training_loss':float(result.training_loss),'train_runtime_seconds':float(result.metrics.get('train_runtime',0)), 'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(0),'note':'GPU smoke training; not evidence of answer quality.'},indent=2))
    (output/'trainer_log_history.json').write_text(json.dumps(trainer.state.log_history,indent=2))
    print('Adapter saved:',output)


if __name__=='__main__': main()
