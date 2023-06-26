from pytorch_lightning import LightningModule
from transformers import AutoModel
import torch
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import f1_score

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
        return x.squeeze(1), self.sig(x)
        

    def training_step(self, batch, batch_idx):
        
        opt_classifier = self.optimizers()
        sentences, label, label_fine = batch
        outputs, o_sig = self.forward(sentences["input_ids"], sentences["attention_mask"], sentences["token_type_ids"])
        
        loss_fine = self.loss(outputs,label_fine)
        #loss_coarse = self.loss(outputs, label)
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
            
        outputs, o_sig = self.forward(sentences["input_ids"], sentences["attention_mask"], sentences["token_type_ids"])
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
        optimizer = AdamW(self.parameters(), lr=0.00002)
        #optimizer_classifier = Adam(self.parameters(), lr=0.001)
        return [optimizer]#, optimizer]
