import json

from src.inference import load_inference_model, predict_response
from src.evaluation import parse_response


def main(config_path, adapter_dir, sentence):
    with open(config_path,'r',encoding='utf-8') as f:
        config = json.load(f)
    lora_model,tokenizer = load_inference_model(config,adapter_dir)
    result = predict_response(sentence,config["system_prompt"],lora_model,tokenizer,max_new_tokens=config["generation"]["max_new_tokens"],repetition_penalty=config["generation"]["repetition_penalty"])
    is_valid,entities = parse_response(sentence,result["response"])

    print(result["response"])
    print(f"{is_valid=}")
    print(f"{entities=}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="中文命名实体抽取")
    parser.add_argument("--config",default="configs/inference.json")
    parser.add_argument("--adapter_path",required=True)
    parser.add_argument("--sentence",required=True)

    args = parser.parse_args()
    main(args.config,args.adapter_path,args.sentence)