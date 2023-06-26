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

sentences, senses = read_dataset("data/coarse-grained/test_coarse_grained.json")
model = build_model("cpu")
for i,sentence in enumerate(sentences):
    print(model.predict(sentence))
    print(senses[i])
    input(".....")