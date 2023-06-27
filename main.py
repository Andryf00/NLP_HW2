from datetime import datetime
from typing import Optional
from Data_module import NLPDataModule
#import datasets
import torch
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from torch import nn
import gc
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from evaluate import *




class WSDTransformer(LightningModule):
    def __init__(
        self,
        model_name: str,
        coarse_training: bool,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.coarse_training = coarse_training

        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last 4 layers of the BERT model
        for param in self.model.encoder.layer[-4:].parameters():
            param.requires_grad = True

        self.linear1 = nn.Linear(in_features=768, out_features=256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=256, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=1)
        self.dropout = nn.Dropout(0.5)
        self.sig = torch.nn.Sigmoid()

        self.classifier = nn.Linear(768,1)
        
        #self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, 
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,    
                ):
        
        model_kwargs = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask,
          "token_type_ids": token_type_ids,
        }
        x = self.model(input_ids, attention_mask, token_type_ids)["pooler_output"]
        X = self.dropout(x)
        x = self.classifier(x)
        """
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear3(x)"""
        return x.squeeze(1), self.sig(x)
        

    def training_step(self, batch, batch_idx):
        opt_classifier = self.optimizers()
        sentences, label, label_fine = batch
        outputs, o_sig = self.forward(sentences["input_ids"], sentences["attention_mask"], sentences["token_type_ids"])
        if self.coarse_training:
            loss_coarse = self.loss(outputs, label)
        else:
            loss_fine = self.loss(outputs,label_fine)
        loss =  loss_fine
        opt_classifier.zero_grad()
        self.manual_backward(loss)
        opt_classifier.step()
        return {"loss":torch.sum(loss)/label.size()[0],
                "pred": [int(o>0.5) for o in o_sig],
                "label": label,
                "label_fine": label_fine}
    
    def training_epoch_end(self, outputs):
        loss = 0
        preds = []
        labels = []
        labels_fine = []

        for o in outputs:
            loss += o["loss"]
            preds+=o["pred"]
            labels+=list(o["label"].cpu().numpy())
            labels_fine+=list(o["label_fine"].cpu().numpy())
        print("\n TRAINING \n",loss/len(outputs),"F1_SCORE_COARSE:", f1_score(y_pred=preds, y_true=labels), "F1_SCORE_FINE:", f1_score(y_pred=preds, y_true=labels_fine))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sentences, label, label_fine = batch
            
        #print(label.size())
        #print(sentences.size())
        outputs, o_sig = self.forward(sentences["input_ids"], sentences["attention_mask"], sentences["token_type_ids"])
        #print(outputs.size())
        w = torch.tensor([5.0 if l==1 else 1.0 for l in label_fine], device=self.device)
        #loss = torch.mean(self.loss(outputs, label_fine) * w)
        loss = self.loss(outputs, label)
        return {"loss": torch.sum(loss)/label.size()[0],
                "pred": [int(o>0.5) for o in o_sig],
                "label": label,
                "label_fine": label_fine}

    def validation_epoch_end(self, outputs):
        if self.trainer.sanity_checking: return
        loss = 0
        preds = []
        labels = []
        labels_fine = []

        for o in outputs:
            loss += o["loss"]
            preds+=o["pred"]
            labels+=list(o["label"].cpu().numpy())
            labels_fine+=list(o["label_fine"].cpu().numpy())
        f1_coarse = f1_score(y_pred=preds, y_true=labels)
        print("\n VALIDATION \n",loss/len(outputs),"F1_SCORE_COARSE:", f1_coarse, "F1_SCORE_FINE:", f1_score(y_pred=preds, y_true=labels_fine))
        self.log_dict({"f1_score":f1_coarse})
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.parameters(), lr=0.00002)
        #optimizer_classifier = Adam(self.parameters(), lr=0.001)
        return [optimizer]#, optimizer]
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = WSDTransformer(checkpoint, coarse_training=False)
    #model = model.load_from_checkpoint("c:\\Users\\andre\\Desktop\\NLP\\nlp2023-hw2\\hw2\\stud\\model\\best_93_72.ckpt")

    early_stopping = EarlyStopping(monitor='f1_score',
                                   mode='max',
                                   patience=3)
    
    checkpoint_callback = ModelCheckpoint(dirpath = "",
                                        save_top_k = 1,
                                        monitor = "f1_score",
                                        mode = "max",
                                        filename = 'best_f1_score',
                                        )

    trainer = pl.Trainer(
        #deterministic = True,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=10,
        default_root_dir="",
        precision=16,
        callbacks=[checkpoint_callback, early_stopping]
    )
    dm = NLPDataModule(tokenizer, batch_size=24, add_example = True)

    model.train()
    trainer.fit(model, dm)
    