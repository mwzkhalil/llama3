import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
from datasets import load_dataset
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer

torch.manual_seed(42)

@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    train_batch_size: Optional[int] = field(default=1)
    eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=17)
    learning_rate: Optional[float] = field(default=3e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.01)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.0)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=256)
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B",
                                       metadata={"help": "The model to train from the Hugging Face hub."})
    #dataset_name: Optional[str] = field(default="iamtarun/python_code_instructions_18k_alpaca",
    #                                    metadata={"help": "The preference dataset to use."})
    use_4bit: Optional[bool] = field(default=True, metadata={"help": "Activate 4bit precision base model loading"})
    use_nested_quant: Optional[bool] = field(default=False, metadata={"help": "Activate nested quantization for 4bit base models"})
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16", metadata={"help": "Compute dtype for 4bit base models"})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "Quantization type fp4 or nf4"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "Number of training epochs for the reward model."})
    fp16: Optional[bool] = field(default=False, metadata={"help": "Enable fp16 training."})
    bf16: Optional[bool] = field(default=True, metadata={"help": "Enable bf16 training."})
    packing: Optional[bool] = field(default=True, metadata={"help": "Use packing dataset creating."})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "Enable gradient checkpointing."})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "The optimizer to use."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate schedule."})
    max_steps: int = field(default=10000000000, metadata={"help": "How many optimizer update steps to take"})
    warmup_steps: int = field(default=100, metadata={"help": "# of steps to do a warmup for"})
    group_by_length: bool = field(default=True, metadata={"help": "Group sequences into batches with same length."})
    save_steps: int = field(default=200, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=5, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(default=False, metadata={"help": "Merge and push weights after training"})
    output_dir: str = field(default="./results_packing", metadata={"help": "Output directory for model predictions and checkpoints."})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def gen_batches_train():
    #ds = load_dataset(script_args.dataset_name, streaming=True, split="train")

    df = pd.read_csv("CortexData.csv")

    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        instruction = row['prompt']
        #input_text = row['input'] if 'input' in row else ""
        out_text = row['response']

        formatted_prompt = (f"user\n\n"
                            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                            f"### Instruction:\n{instruction}\n\n"
                            f"### Response:\n"
                            f"assistant\n\n"
                            f"{out_text}\n")

        yield {'text': formatted_prompt}


def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        use_auth_token=True,
    )

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj'],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


training_arguments = TrainingArguments(
    num_train_epochs=script_args.num_train_epochs,
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    warmup_steps=script_args.warmup_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    report_to='wandb',
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)

train_gen = Dataset.from_generator(gen_batches_train)

tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    train_dataset=train_gen,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.train()

if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)

    del model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
