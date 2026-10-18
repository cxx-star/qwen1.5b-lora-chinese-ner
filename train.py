import json
import torch
from transformers import set_seed
from src.data import load_samples,load_split,prepare_training_sample,collate_samples
from src.training import load_training_model,train_one_epoch_plain,save_checkpoint
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path

def main(training_config_path, data_path, split_path, output_dir):
    with open(training_config_path,'r',encoding='utf-8') as f:
        training_config = json.load(f)
    with open(training_config["model_config"],'r',encoding='utf-8') as f:
        model_config = json.load(f)
    set_seed(training_config["seed"])
    samples = load_samples(data_path)
    train_samples,val_samples = load_split(samples,split_path)
    lora_model,tokenizer = load_training_model(model_config,training_config)
    encoded_samples = [prepare_training_sample(sample,model_config["system_prompt"],tokenizer) for sample in train_samples]
    collate_fn = partial(collate_samples,pad_token_id=tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(training_config["seed"])
    train_dataloader = DataLoader(encoded_samples,batch_size=training_config["batch_size"],shuffle=True,collate_fn=collate_fn,generator=generator)
    lora_optimizer = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad],lr=training_config["lr"],weight_decay=training_config["weight_decay"])
    for i in range(1,training_config["epochs"] + 1):
        epoch_loss,update_steps = train_one_epoch_plain(lora_model,train_dataloader,lora_optimizer)
        print("epoch:",i,"loss:",epoch_loss)
        print("steps:",update_steps)
        save_checkpoint(lora_model,tokenizer,lora_optimizer,Path(output_dir) / f"epoch{i}")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练中文实体抽取 LoRA")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--data", default="data/raw/train.json")
    parser.add_argument("--split", default="data/splits/train_val_indices.json")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main(args.config, args.data, args.split, args.output_dir)
