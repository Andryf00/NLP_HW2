import numpy as np
from typing import List, Dict

from hw2.stud.wsd_transformer import WSDTransformer
from transformers import AutoTokenizer
import json

def build_model(device: str) -> WSDTransformer:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)


class RandomBaseline():

    def __init__(self):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        pass

    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        return [[np.random.choice(candidates) for candidates in sentence_data["candidates"].values()]
                for sentence_data in sentences]


class StudentModel():

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        checkpoint = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = WSDTransformer(checkpoint)
        self.model.device = device
        self.model = self.model.load_from_checkpoint("hw2\\model\\fine_trained.ckpt")
        self.model.eval()
        self.coarse_to_fine = self.load_mapping(())

    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        list_output=[]
        for sample in sentences:
            for idx in list(sample["instance_ids"].keys()):
                list_sentence = []
                dict={}
                idx = int(idx)
                sentence = sample["words"]
                #print("index of w", idx, sentence[idx])
                target = sentence[idx] 
                sentence[idx] = "\""+target+"\""
                curr_max = 0
                for sense in sample["candidates"][str(idx)]:
                    for fine_sense_dict in list(self.coarse_to_fine[sense]):
                        fine_sense = list(fine_sense_dict.values())
                        sense_key = list(fine_sense_dict.keys())[0]
                        sentence_gloss = " ".join(sentence + ["[SEP]"] + [target, ":"] + fine_sense)

                        
                        tokenized = self.tokenizer([sentence_gloss],
                                            return_tensors="pt")
                        _, output = self.model(**tokenized)
                        if output>curr_max: 
                            curr_max = output
                            current_prediction = sense
                list_sentence.append(current_prediction)
            list_output.append(list_sentence)
    
        return list_output
        
    def load_mapping():
        with open("data\\map\\coarse_fine_defs_map.json") as f:
            data = json.load(f)
        coarse_senses=list(data.keys())
        dict = {}
        for sense in coarse_senses:
            f = data[sense]
            dict[sense] = f
        return dict
