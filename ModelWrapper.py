import pytorch_lightning as pl
import torch.nn as nn
import torch
from Metrics import Metrics
import torchvision.transforms as T

mean=torch.tensor([84.91024075/255, 84.17390113/255, 75.42844544/255])
std=torch.tensor([49.44622809/255, 47.72316675/255, 47.68987178/255])
std_inv = 1 / (std + 1e-7)
unnormalize = T.Normalize(-mean * std_inv, std_inv)


class ModelWrapper(pl.LightningModule):
        def __init__(self, model,loss):
            super().__init__()
            self.loss_func = loss
            self.model = model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_func(y_hat, y)
            y_hat = (y_hat>0.5).bool() # Switch from probabilities to actual choice
            f1 = Metrics.F1(y.bool(),y_hat)
            self.logger.experiment.add_scalars('Loss', {'Train loss': loss}, global_step=self.global_step)
            self.logger.experiment.add_scalars('F1 score', {'Train F1': f1}, global_step=self.global_step)
            return loss

        def predict_step(self, batch, batch_idx):
            if(len(batch)==2): #If the is gt
              image, gt = batch
              return self.model(image),unnormalize(image),gt
            return self.model(batch),unnormalize(batch)

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)
            y_hat = (y_hat>0.5).bool() # Switch from probabilities to actual choice
            f1 = Metrics.F1(y.bool(),y_hat)
            self.log("val_F1",f1,batch_size=16)
            self.logger.experiment.add_scalars('Loss', {'Val loss': loss}, global_step=self.global_step)
            self.logger.experiment.add_scalars('F1 score', {'Val F1': f1}, global_step=self.global_step)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.01)
