from torchvision.transforms import v2
from data.transforms.transform import TransposeTransform, EEGFourierTransform
import wandb
import torch
from torch.utils.data import random_split, DataLoader
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import torchaudio
from scipy.signal import butter, lfilter
from streaming import StreamingDataset, StreamingDataLoader
import torch
import torchaudio
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import os


wandb.login()
output_path = '/media/elijah/T7/ds117_streaming_eeg/'

batch_size = 32
dataset = StreamingDataset(local=output_path, batch_size=batch_size)
loader = StreamingDataLoader(dataset,batch_size=batch_size)


def compute_spectrograms(eeg_data, fs=256, n_fft=256, hop_length=128):
    batch_size, sequence_length, num_channels = eeg_data.shape
    spectrogram_images = []

    for batch_idx in range(batch_size):
        spectrograms = []
        for channel_idx in range(num_channels):

            channel_data = eeg_data[batch_idx, :, channel_idx]
            spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(channel_data)
            spectrograms.append(spec)
        spectrogram_images.append(torch.stack(spectrograms, dim=0))

    return torch.stack(spectrogram_images, dim=0)


class EEGStepClassifier(pl.LightningModule):
    def __init__(self, num_channels=74, d_model=512, num_classes=8, fs=256, n_fft=256, hop_length=128):
        super().__init__()
        self.save_hyperparameters()
        self.ignore_index = -1
        self.fs = fs
        self.a = 1.1
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.embedding = nn.Embedding(num_embeddings=17, embedding_dim=d_model)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        )

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(4096, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def compute_spectrograms(self, eeg_data):
        batch_size, sequence_length, num_channels = eeg_data.shape
        spectrogram_images = []

        for batch_idx in range(batch_size):
            spectrograms = []
            for channel_idx in range(num_channels):
                channel_data = eeg_data[batch_idx, :, channel_idx]
                spec_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length).to(eeg_data.device)
                spec = spec_transform(channel_data)
                spectrograms.append(spec)
            spectrogram_images.append(torch.stack(spectrograms, dim=0))

        return torch.stack(spectrogram_images, dim=0)

    def forward(self, x, id):
        x = torch.tensor([self.bandpass_filter(channel.cpu().numpy(), 0.5, 100, self.fs) for channel in x.permute(0, 2, 1)], device=x.device).permute(0, 2, 1).float()
        
        x = self.compute_spectrograms(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        batch_size, num_channels, freq_bins, time_bins = x.shape
        x = x.view(batch_size, -1)
        
        ids = torch.tensor([self.get_id(iid) for iid in id], device=self.device)
        user_emb = self.embedding(ids)
        
        x = self.dropout(self.fc1(x))
        x = torch.cat((x, user_emb), dim=-1)
        x = self.mlp(x)
        logits = self.classifier(x)
        return logits

    def get_id(self, name: str):
        return int(name) if name.isdigit() else 0

    def training_step(self, batch, batch_idx):
        x = batch['eeg'].permute(0, 2, 1)
        y = batch['label'][..., -1].long()
        id = batch['subject']
        logits = self.forward(x, id)
        loss = F.cross_entropy(logits, y, ignore_index=self.ignore_index)
        mask = y != self.ignore_index
        acc = self.accuracy(logits[mask], y[mask])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['eeg'].permute(0, 2, 1)
        y = batch['label'].view(-1)
        id = batch['subject']
        logits = self.forward(x, id)
        loss = F.cross_entropy(logits, y, ignore_index=self.ignore_index)
        mask = y != self.ignore_index
        acc = self.accuracy(logits[mask], y[mask])
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, threshold=1e-2),
            'monitor': 'train_acc' 
        }
        return [optimizer], [scheduler]


n_classes = 3
model = EEGStepClassifier(num_classes=n_classes, d_model=256)

logger = WandbLogger(project='diploma', name='diploma_eeg_spectr_conv',)
trainer = Trainer(max_epochs=50, log_every_n_steps=1, logger=logger, )

trainer.fit(model, loader)
