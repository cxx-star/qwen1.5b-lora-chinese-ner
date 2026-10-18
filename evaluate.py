import json

from src.evaluation import evaluate_counts, compute_matrix


def main(predictions_path):
    samples = []
    with open(predictions_path,'r',encoding='utf-8') as f:
        line = f.readline()
        while line:
            sample = json.loads(line)
            samples.append(sample)

            line = f.readline()
    counts = evaluate_counts(samples)
    matrics = compute_matrix(*counts[:3])
    print("length:",len(samples))
    print("counts:",counts)
    print("matrics:",matrics)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="重新评估已保存的实体抽取预测")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    main(args.predictions)