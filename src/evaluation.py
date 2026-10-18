import json

def check_entity(sentence,entity):
    if isinstance(entity,dict):
        allowed_types = [
                        "address", "book", "company", "game", "government",
                        "movie", "name", "organization", "position", "scene"
                        ]
        required_fields = ["type", "text", "start", "end"]
        for field in required_fields:
            if field not in entity:
                return False
        if entity["type"] not in allowed_types:
            return False
        start = entity["start"]
        end = entity["end"]
        if type(start) is not int or type(end) is not int:
            return False
        name = sentence[start:end]
        position_valid = 0 <= start <end <= len(sentence)
        return name == entity["text"] and position_valid
    else:
        return False

def check_entities(sentence,entities):
    for entity in entities:
        if not check_entity(sentence,entity):
            return False
    return True
def parse_response(sentence,response):
    try:
        parsed_entities = json.loads(response)
        if isinstance(parsed_entities,list):
            if not check_entities(sentence,parsed_entities):
                return False,[]
        else:
            print("输出必须是 JSON 数组")
            return False,[]
    except json.JSONDecodeError:
        print("JSON 格式错误")
        return False,[]
    return True,parsed_entities
def entity_set(entities):
    e = set()
    for entity in entities:
        e.add((entity["type"],entity["start"],entity["end"]))
    return e

def count_matches(gold_entities,pred_entities):
    gold_entity_set = entity_set(gold_entities)
    pred_entity_set = entity_set(pred_entities)
    tp = len(gold_entity_set & pred_entity_set)
    fp = len(pred_entity_set - gold_entity_set)
    fn = len(gold_entity_set - pred_entity_set)
    return tp,fp,fn

def compute_matrix(tp,fp,fn):
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    return precision,recall,f1

def evaluate_counts(samples):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    invalid_counts = 0
    for sample in samples:
        is_valid,pred_entities = parse_response(sample["sentence"],sample["response"])
        if not is_valid:
            invalid_counts += 1
        tp,fp,fn = count_matches(sample["gold"],pred_entities)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return total_tp,total_fp,total_fn,invalid_counts