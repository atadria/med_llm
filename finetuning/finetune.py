import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_from_disk, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# Load model

base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                             quantization_config=bnb_config)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=4096,
    padding_side="left",
    add_eos_token=True)

# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.unk_token
model.config.pad_token_id = tokenizer.pad_token

# Data preparation
#
dataset = concatenate_datasets([
    load_from_disk('data/simple_text_med_dataset.hf'),
    load_from_disk('data/instruct_med_dataset.hf')
])

dataset = dataset.shuffle()


# Prepare for training
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


config = LoraConfig(
    r=128,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

if torch.cuda.device_count() > 1:  # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True
    print('### multiple GPU ###')

# Training
output_dir = "./med_mistral"
tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_steps=5,
        num_train_epochs=2,
        #max_steps=20,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to='none',  # or log to WanDB
        logging_dir="./logs",
    ),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# Save model
model.save_pretrained("med_mistral")
tokenizer.save_pretrained("med_mistral")

model = model.merge_and_unload()
model.push_to_hub("adriata/med_mistral", use_temp_dir=False, token="")
tokenizer.push_to_hub("adriata/med_mistral", use_temp_dir=False, token="")
