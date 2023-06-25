
from cProfile import label
from random import shuffle
import pytorch_lightning as pl
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import torch



"""with open("C:\\Users\\andre\\Desktop\\NLP\\nlp2023-hw2\\data\\coarse-grained\\" + "train_coarse_grained.json") as f:
    data = json.load(f)
sentences=list(data.keys())
#print(data[sentences[0]])
data_ = []


keys=list(data[sentences[0]].keys())
#print(keys)"""

class NLPDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_dir: str = "path/to/dir", batch_size: int = 64):
        super().__init__()
        self.coarse_data_dir = "C:\\Users\\andre\\Desktop\\NLP\\nlp2023-hw2\\data\\coarse-grained\\"
        self.fine_data_dir = "C:\\Users\\andre\\Desktop\\NLP\\nlp2023-hw2\\data\\fine-grained\\"
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def setup(self, stage: str): 
        self.coarse_to_fine = self.load_mapping("C:\\Users\\andre\\Desktop\\NLP\\nlp2023-hw2\\data\\map\\")
        self.train_data = self.load_from_json(self.coarse_data_dir + "train_coarse_grained.json", self.fine_data_dir + "train_fine_grained.json")
        self.test_data = self.load_from_json(self.coarse_data_dir + "test_coarse_grained.json", self.fine_data_dir + "test_fine_grained.json")
        self.val_data = self.load_from_json(self.coarse_data_dir + "val_coarse_grained.json", self.fine_data_dir + "val_fine_grained.json")
        

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          collate_fn = self.custom_collate,
                          num_workers=0,
                          pin_memory=False
                          )

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn = self.custom_collate,
                          num_workers=0,
                          pin_memory=False
                          )

    def test_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          collate_fn = self.custom_collate,
                          num_workers=0,
                          pin_memory=False
                        )

    def load_from_json(self, path_coarse, path_fine):
        with open(path_coarse) as f:
            data_coarse = json.load(f)
        with open(path_fine) as f:
            data_fine = json.load(f)
        sentences=list(data_coarse.keys())
        lenghts = []
        keys = ['instance_ids', 'lemmas', 'pos_tags', 'senses', 'words', 'candidates']
        dataset = []
        i=0
        for id in tqdm(sentences):
            #print("id:", id)
            i+=1
            #if i == 50:break
            sample = data_coarse[id]
            sample_fine = data_fine[id]
            #print("sample:", sample)
            dict={}
            for idx in list(sample["instance_ids"].keys()):
                idx = int(idx)
                sentence = sample["words"]
                #print("index of w", idx, sentence[idx])
                target = sentence[idx] 
                sentence[idx] = "\""+target+"\""
                for sense in sample["candidates"][str(idx)]:
                    for fine_sense in list(self.coarse_to_fine[sense]):
                        #print(self.coarse_to_fine[sense])
                        value = list(fine_sense.values())
                        #print(value)
                        key = list(fine_sense.keys())[0]
                        #print(key)
                        #lenghts.append(len(sentence))
                        sentence_gloss = " ".join(sentence + ["[SEP]"] + [target, ":"] + value)
                        #print(sense, self.coarse_to_fine[sense])
                        #print(sample["senses"][str(idx)][0])
                        #print(sense)
                        label = float(sample["senses"][str(idx)][0] == sense)
                        label_fine = float(key == sample_fine["senses"][str(idx)][0])
                        #print("final sentence:", " ".join(sentence_gloss), label)

                        dataset.append((sentence_gloss , label, label_fine))
                        #print(sentence_gloss, label, label_fine)
                        #print(sample_fine["senses"][str(idx)][0], key)
                        #print(sample["senses"][str(idx)][0], sense)
                        #input("------------")
        """import matplotlib.pyplot as plt
        plt.hist(lenghts)
        plt.show()
        """
        return dataset
        
    def load_mapping(self, path):
        with open(path + "coarse_fine_defs_map.json") as f:
            data = json.load(f)
        coarse_senses=list(data.keys())
        dict = {}
        for sense in coarse_senses:

            f = data[sense]
            #print(f)
            #input("--")
            fine_senses = []
            f#for fi in f:
            #    fine_senses += list(fi)
            dict[sense] = f
        return dict
    
    def custom_collate(self, batch):
        batch_out = self.tokenizer([sentence for sentence, label, label_fine in batch],
                                    padding=True,
                                    truncation = True,
                                    return_tensors="pt")
        """for s in batch_out["input_ids"]:
            print(self.tokenizer.decode(s))
            input("...")
        """
        labels = torch.HalfTensor([label for sentence, label, label_fine in batch])
        labels_fine = torch.HalfTensor([label_fine for sentence, label, label_fine in batch])
        return batch_out, labels, labels_fine
    
