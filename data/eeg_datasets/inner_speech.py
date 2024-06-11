import os
import mne
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class BIDSEEGDataset(Dataset):
    def __init__(self, bids_path, transform=None, resample_rate=None, fixed_length=2048):
        self.bids_path = bids_path
        self.transform = transform
        self.resample_rate = resample_rate
        self.fixed_length = fixed_length
        self.encoder = LabelEncoder()
        self.samples = []

        self.file_info = self._load_file_info()
        if 'label' in self.file_info.columns:
            self.file_info['label'] = self.encoder.fit_transform(self.file_info['label'])

    def _load_file_info(self):
        file_data = []
        for subject_dir in os.listdir(self.bids_path):
            full_subject_dir = os.path.join(self.bids_path, subject_dir)
            if subject_dir.startswith('sub-'):
                file_data.extend(self._load_subject_data(full_subject_dir))
        return pd.DataFrame(file_data)

    def _load_subject_data(self, subject_dir):
        session_data = []
        for session_dir in os.listdir(subject_dir):
            if session_dir.startswith('ses-meg'):
                meg_dir = os.path.join(subject_dir, session_dir, 'meg')
                for file in os.listdir(meg_dir):
                    if file.endswith('_meg.fif'):
                        fif_path = os.path.join(meg_dir, file)
                        events_path = fif_path.replace('_meg.fif', '_events.tsv')
                        if os.path.exists(events_path):
                            events_df = pd.read_csv(events_path, sep='\t')
                            onsets = events_df['onset'].values
                            durations = np.diff(onsets, append=onsets[-1] + 1)  # default_duration is assumed to be 1 if not given
                            session_data.append({
                                'fif_path': fif_path,
                                'start': onsets[0],
                                'duration': durations[0],
                                'label': events_df['stim_type'].iloc[0]
                            })
        return session_data

    def _preprocess_single_item(self, item):
        raw = mne.io.read_raw_fif(item['fif_path'], preload=True, verbose=False)
        raw.pick_types(eeg=True, meg=False, verbose=False)
        
        if self.resample_rate:
            raw.resample(self.resample_rate)

        sfreq = raw.info['sfreq']
        start_sample = int(item['start'] * sfreq)
        desired_samples = self.fixed_length

        if raw.n_times < start_sample + desired_samples:
            eeg_segment = raw.get_data(start=start_sample)
            # Padding if necessary
            pad_size = desired_samples - eeg_segment.shape[1]
            eeg_segment = np.pad(eeg_segment, ((0, 0), (0, pad_size)), mode='constant')
        else:
            eeg_segment = raw.get_data(start=start_sample, stop=start_sample + desired_samples)

        eeg_tensor = torch.from_numpy(eeg_segment).float()
        labels_tensor = torch.full((desired_samples,), item['label'], dtype=torch.long)

        if self.transform:
            eeg_tensor = self.transform(eeg_tensor)

        return eeg_tensor, labels_tensor

    def __len__(self):
        return len(self.file_info)

    def __getitem__(self, idx):
        item = self.file_info.iloc[idx]
        eeg_tensor, labels_tensor = self._preprocess_single_item(item)
        return eeg_tensor, labels_tensor
