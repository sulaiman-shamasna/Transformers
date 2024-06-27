from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
import torch

def perform_inference(model_path, prompt):
    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model_inf = PeftModel.from_pretrained(model, model_path)

    encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_config = model_inf.generation_config
    gen_config.max_new_tokens = 200
    gen_config.temperature = 0.2
    gen_config.top_p = 0.7
    gen_config.num_return_sequences = 1
    gen_config.pad_token_id = tokenizer.eos_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask, generation_config=gen_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
