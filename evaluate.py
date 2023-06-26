import json
from tqdm import tqdm

def eval(model, tokenizer):
    with open("C:\\Users\\andre\\Desktop\\NLP\\nlp2023-hw2\\data\\coarse-grained\\test_coarse_grained.json") as f:
        data_coarse = json.load(f)
    with open("C:\\Users\\andre\\Desktop\\NLP\\nlp2023-hw2\\data\\fine-grained\\test_fine_grained.json") as f:
        data_fine = json.load(f)
    model.eval()
    sentences=list(data_coarse.keys())
    trues = totals =0
    coarse_to_fine = load_mapping("C:\\Users\\andre\\Desktop\\NLP\\nlp2023-hw2\\data\\map\\")
    for id in tqdm(sentences):
        sample = data_coarse[id]
        sample_fine = data_fine[id]
        for idx in list(sample["instance_ids"].keys()):
            dict={}
            idx = int(idx)
            sentence = sample["words"]
            #print("index of w", idx, sentence[idx])
            target = sentence[idx] 
            sentence[idx] = "\""+target+"\""
            for sense in sample["candidates"][str(idx)]:
                for fine_sense in list(coarse_to_fine[sense]):
                    value = list(fine_sense.values())
                    key = list(fine_sense.keys())[0]
                    sentence_gloss = " ".join(sentence + ["[SEP]"] + [target, ":"] + value)

                    
                    tokenized = tokenizer([sentence_gloss],
                                        return_tensors="pt")
                    _, output = model(**tokenized)
                    dict[output] = key
            logit_max = max(list(dict.keys()))
            """print(logit_max)
            print(dict[logit_max])
            print([sample["senses"][str(idx)][0]])
            print(coarse_to_fine[sample["senses"][str(idx)][0]])
            print([list(sense_dict.keys()) for sense_dict in coarse_to_fine[sample["senses"][str(idx)][0]]])
            input("...")
            """
            if dict[logit_max] in flatten([list(sense_dict.keys()) for sense_dict in coarse_to_fine[sample["senses"][str(idx)][0]]]):
                trues+=1
            #print(trues, totals)
            totals += 1
            if totals%100==0:
                print(trues, "/", totals, trues/totals)
                
    print(trues, "/", totals, trues/totals)
    
    
def load_mapping(path):
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

def flatten(l):
    return [item for sublist in l for item in sublist]