import torch
import json
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
def predict_response(sentence,system_prompt,inference_model,tokenizer,max_new_tokens=256,example=None,repetition_penalty = 1.0):
    inference_model.eval()
    messages = [{"role":"system","content":system_prompt}]
    if example is not None:
        messages.append({"role":"user","content":example["sentence"]})
        messages.append({"role":"assistant","content":json.dumps(example["gold"],ensure_ascii=False)})
    messages.append({"role":"user","content":sentence})
    inputs = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt").to(inference_model.device)
    with torch.inference_mode():
        output_ids = inference_model.generate(**inputs,do_sample = False,max_new_tokens=max_new_tokens,use_cache=True,repetition_penalty=repetition_penalty)
    input_length = inputs["input_ids"].shape[1]
    new_ids = output_ids[0,input_length:]
    raw_response = tokenizer.decode(new_ids,skip_special_tokens=True)
    eos_ids = inference_model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids,int):
        eos_ids = [eos_ids]
    ended_with_EOS = (len(new_ids) > 0 and new_ids[-1].item() in eos_ids)
    return {"response":raw_response,"input_tokens":input_length,"generated_tokens":len(new_ids),"ended_with_EOS":ended_with_EOS,"hit_length_limit":len(new_ids) == max_new_tokens}


def load_inference_model(config, adapter_dir):
    model_id = config["model_id"]
    revision = config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id,revision=revision)
    model_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id,revision=revision,dtype = model_dtype,device_map="cuda:0")
    lora_model = PeftModel.from_pretrained(model,adapter_dir,is_trainable = False)
    return lora_model,tokenizer
