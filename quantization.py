# https://github.com/casper-hansen/AutoAWQ/tree/main/examples

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import HfApi


model_path = 'adriata/med_mistral'
quant_path = 'adriata/med_mistral_awq'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')

# upload model
api = HfApi()

api.upload_folder(
    folder_path=quant_path,
    repo_id=quant_path,
    repo_type="model",
    token="",
    commit_message='upload model'
)
