import wandb
import numpy as np
from streaming import StreamingDataset, StreamingDataLoader
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


wandb.login()
output_path='/media/elijah/T7/ds000117-mri-streaming/'
batch_size = 1
dataset = StreamingDataset(local=output_path, batch_size=batch_size)
loader = StreamingDataLoader(dataset,batch_size=batch_size)

class FMRIVolumeClassifier(pl.LightningModule):
    def __init__(self, num_subjects, d_model=512, num_classes=10, input_shape=(1, 64, 64, 33)):
        super().__init__()
        self.save_hyperparameters()
      
        self.embedding = nn.Embedding(num_embeddings=num_subjects, embedding_dim=d_model)
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(64),
        )
        
        conv3d_output_size = self._get_conv3d_output_size(input_shape)
        self.fc_input_size = conv3d_output_size * 64
        
        self.encoder = nn.Sequential(
            nn.Linear(self.fc_input_size, d_model),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
        )

        self.adapter = nn.Sequential(nn.Linear(2 * d_model, d_model),
                                     nn.LeakyReLU(0.01),
                                     nn.Linear(d_model, d_model),
                                     nn.LeakyReLU(0.01),
                                     )
        
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def _get_conv3d_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.conv3d(dummy_input)
            return torch.tensor(output.shape[2:]).prod().item()

    def forward(self, x, subject_id):
        x = x.view(x.shape[0], x.shape[4], 1, x.shape[1], x.shape[2], x.shape[3])
        batch_size, seq_len, _, _, _, _ = x.size()
        x = x.view(batch_size * seq_len, *x.size()[2:])
        x = self.conv3d(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.encoder(x)
        x = self.adapter(torch.cat([x, self.embedding(subject_id)], dim=-1))

        x, _ = self.lstm(x)
        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        x = batch['func']
        y = batch['label'].long()
        subject_id = batch['subject']
        logits = self.forward(x, subject_id)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        acc = self.accuracy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['func']
        y = batch['label']
        subject_id = batch['subject']
        logits = self.forward(x, subject_id)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        acc = self.accuracy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
        return [optimizer], [scheduler]


n_classes = 4
model = FMRIVolumeClassifier(num_classes=n_classes,
                               num_subjects=17,
                               d_model=128,
                            )


logger = WandbLogger(project='diploma', name='diploma_fmri_LSTM3dConv',)

trainer = Trainer(max_epochs=90,
                  gradient_clip_val=0.7,
                  log_every_n_steps=1,
                  default_root_dir="my_checkopoints/fmri_3d_conv/",
                  enable_checkpointing=True,
                  precision='16-mixed',
                  detect_anomaly=True,
                  accumulate_grad_batches=8)

trainer.fit(model, loader)