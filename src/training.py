import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig,get_peft_model
from pathlib import Path


def train_one_epoch_plain(model, loader, optimizer):
    epoch_loss = 0
    update_steps = 0
    model.train()
    model.config.use_cache = False
    total_tokens = 0
    for batch in loader:
        train_batch = {key:value.to(model.device) for key,value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            train_outputs = model(input_ids = train_batch["input_ids"],attention_mask=train_batch["attention_mask"],labels = train_batch["labels"],use_cache=False)
            loss = train_outputs.loss
        loss.backward()
        optimizer.step()
        tokens = (train_batch["labels"][:,1:] != -100).sum().item()
        total_tokens += tokens
        epoch_loss += loss.item()*tokens
        update_steps += 1
        del train_outputs, loss
    epoch_loss = epoch_loss / total_tokens
    return epoch_loss, update_steps

def load_training_model(model_config, training_config):
    if not torch.cuda.is_available():
        raise RuntimeError("当前训练入口需要 CUDA 显卡")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前训练入口需要支持 BF16 的 CUDA 显卡")

    model_id = model_config["model_id"]
    revision = model_config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id,revision=revision)
    model_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id,revision=revision,dtype = model_dtype,device_map="cuda:0")
    lora_config = LoraConfig(**training_config["lora"])
    lora_model = get_peft_model(model,lora_config,revision=revision)
    return lora_model,tokenizer

def save_checkpoint(model,tokenizer,optimizer,save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True,exist_ok=True)

    model.save_pretrained(save_dir/"adapter")
    tokenizer.save_pretrained(save_dir/"tokenizer")
    torch.save(optimizer.state_dict(),save_dir/"optimizer.pt")
