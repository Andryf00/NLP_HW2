from datetime import datetime
from typing import Optional
from Data_module import NLPDataModule
#import datasets
import torch
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from torch import nn
import gc

class Classifier(LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.linear1 = nn.Linear(in_features=768, out_features=256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=256, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class WSDTransformer(LightningModule):
    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False

        self.linear1 = nn.Linear(in_features=768, out_features=256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=256, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=1)
        self.dropout = nn.Dropout(0.5)
        self.sig = torch.nn.Sigmoid()
        
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
        x = self.model(**model_kwargs)["pooler_output"]
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x.squeeze(1), self.sig(x).squeeze(1)
        #print(embeds)
        return x
        

    def training_step(self, batch, batch_idx):
        #opt_classifier, opt_transformer = self.optimizers()
        
        opt_classifier = self.optimizers()
        sentences, label, label_fine = batch
        #print(sentences.size())
        outputs, o_sig = self.forward(sentences["input_ids"], sentences["attention_mask"], sentences["token_type_ids"])
        
        loss_fine = self.loss(outputs, label_fine)
        loss_coarse = self.loss(outputs, label)
        loss = loss_fine + loss_coarse
        #opt_transformer.zero_grad()
        opt_classifier.zero_grad()
        self.manual_backward(loss)
        #opt_transformer.step()
        opt_classifier.step()
        #if batch_idx%4000==0:
            #print(label)
            #print(label_fine)
            #print("OUTPUTS", outputs)
            #print([int(o>0.5) for o in outputs])
            #print("LOSS:",torch.sum(loss)/label.size()[0])
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
        print("\n VALIDATION \n",loss/len(outputs),"F1_SCORE_COARSE:", f1_score(y_pred=preds, y_true=labels), "F1_SCORE_FINE:", f1_score(y_pred=preds, y_true=labels_fine))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        #optimizer = AdamW(self.model.parameters(), lr=0.0001)
        optimizer_classifier = Adam(self.parameters(), lr=0.001)
        return [optimizer_classifier]#, optimizer]
    
if __name__ == '__main__':
    checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = WSDTransformer(checkpoint)
    torch.cuda.empty_cache()
    gc.collect()
    trainer = pl.Trainer(
        #deterministic = True,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=50,
        default_root_dir="",
        precision=16,
        #callbacks=[checkpoint_callback, early_stopping]
    )
    dm = NLPDataModule(tokenizer, batch_size=64)
    trainer.fit(model, dm)