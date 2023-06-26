import json
from typing import Tuple, List, Any, Dict
from hw2.stud.implementation import build_model

def read_dataset(path: str) -> Tuple[List[Dict], List[List[List[str]]]]:
    sentences_s, senses_s = [], []

    with open(path) as f:
        data = json.load(f)

    for sentence_id, sentence_data in data.items():
        assert len(sentence_data["instance_ids"]) > 0
        assert (len(sentence_data["instance_ids"]) ==
                len(sentence_data["senses"]) ==
                len(sentence_data["candidates"]))
        assert all(len(gt) > 0 for gt in sentence_data["senses"].values())
        assert (all(gt_sense in candidates for gt_sense in gt)
                for gt, candidates in zip(sentence_data["senses"].values(), sentence_data["candidates"].values()))
        assert len(sentence_data["words"]) == len(sentence_data["lemmas"]) == len(sentence_data["pos_tags"])
        senses_s.append(list(sentence_data.pop("senses").values()))
        sentence_data["id"] = sentence_id
        sentences_s.append(sentence_data)

    assert len(sentences_s) == len(senses_s)

    return sentences_s, senses_s

def get_n_instances(l: List[List[Any]]) -> int:
    return sum(len(inner_l) for inner_l in l)


def wsd_accuracy_score(senses_s: List[List[List[str]]], predictions_s: List[List[str]]) -> float:
    if len(senses_s) != len(predictions_s):
        raise ValueError(f"The number of input sents and the number of sents returned in predictions do not match: # "
                         f"input sents = {len(senses_s)}, # returned sents = {len(predictions_s)}")
    n_instances = get_n_instances(senses_s)
    print(f"# instances: {n_instances}")
    correct = 0
    for i, (senses, predictions) in enumerate(zip(senses_s, predictions_s)):
        if len(senses) != len(predictions):
            raise ValueError(
                f"For the sentence with idx {i}, the number of input WSD instances and the number of WSD instances "
                f"returned in predictions do not match: # input instances = {len(senses)}, # returned instances = "
                f"{len(predictions)}")
        for sense, prediction in zip(senses, predictions):
            if prediction in sense:
                correct += 1
    return correct / n_instances

sentences, senses = read_dataset("data/coarse-grained/test_coarse_grained.json")
model = build_model("cuda")
batch_size = 64
predictions_s = []
for i in range(0, len(sentences), batch_size):
    batch = sentences[i: i + batch_size]
    preds = model.predict(batch)
    #print(model.predict(batch))
    #print(senses[i:i+batch_size])
    #input(".....")
    predictions_s += preds

acc = wsd_accuracy_score(senses, predictions_s)
print(acc)