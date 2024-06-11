import os
import mne
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import LabelEncoder
from streaming import MDSWriter

def load_file_info(bids_path):
    file_data = []
    for subject_dir in os.listdir(bids_path):
        full_subject_dir = os.path.join(bids_path, subject_dir)
        if subject_dir.startswith('sub-'):
            for session_dir in os.listdir(full_subject_dir):
                if session_dir.startswith('ses-meg'):
                    meg_dir = os.path.join(full_subject_dir, session_dir, 'meg')
                    for file in os.listdir(meg_dir):
                        if file.endswith('_meg.fif'):
                            fif_path = os.path.join(meg_dir, file)
                            events_path = fif_path.replace('_meg.fif', '_events.tsv')
                            if os.path.exists(events_path):
                                file_data.append({'subject': subject_dir.split('-')[-1], 'fif_path': fif_path, 'events_path': events_path})
    return pd.DataFrame(file_data)

def load_subject_data(events_path, encoder):
    events_df = pd.read_csv(events_path, sep='\t')
    session_data = []
    for idx, event in events_df.iterrows():
        session_data.append({
            'start': event['onset'],
            'duration': event['duration'] if 'duration' in event else 1.0,  # Assuming default duration as 1.0 if not provided
            'label': event['stim_type'],
        })
    df = pd.DataFrame(session_data)
    df['label'] = encoder.transform(df['label'])
    df['duration'] = np.diff(df['start'], append=df['start'].iloc[-1] + 1)

    return df

def create_label_vector(events, sfreq, start_sample, segment_length):
    label_vector = np.zeros(segment_length, dtype=int)
    for event in events:
        event_start_sample = int(event['start'] * sfreq)
        event_end_sample = event_start_sample + int(event['duration'] * sfreq)
        event_label = event['label']
        
        for i in range(segment_length):
            sample_index = start_sample + i
            if event_start_sample <= sample_index < event_end_sample:
                label_vector[i] = event_label
                
    return label_vector

def preprocess_file(file_info, encoder, fixed_length=2048, resample_rate=None, transform=None):
    raw = mne.io.read_raw_fif(file_info['fif_path'], preload=True, verbose=False)
    raw.pick_types(eeg=True, meg=False, verbose=False)

    if resample_rate:
        raw.resample(resample_rate)

    sfreq = raw.info['sfreq']
    total_samples = raw.n_times
    subject = file_info['subject']

    events = load_subject_data(file_info['events_path'], encoder).to_dict('records')
    # yield events

    for start in range(0, total_samples, fixed_length):
        end = start + fixed_length
        if end > total_samples:
            break  # Ignore the last segment if it's smaller than the fixed length

        eeg_segment = raw.get_data(start=start, stop=end)
        if transform:
            eeg_segment = transform(eeg_segment)


        label_vector = create_label_vector(events, sfreq, start, fixed_length)

        yield {
            'eeg': eeg_segment.astype(np.float32),
            'label': label_vector.astype(np.int32),
            'subject': subject
        }

def fit_label_encoder(bids_path):
    all_labels = []
    for subject_dir in os.listdir(bids_path):
        full_subject_dir = os.path.join(bids_path, subject_dir)
        if subject_dir.startswith('sub-'):
            for session_dir in os.listdir(full_subject_dir):
                if session_dir.startswith('ses-meg'):
                    meg_dir = os.path.join(full_subject_dir, session_dir, 'meg')
                    for file in os.listdir(meg_dir):
                        if file.endswith('_meg.fif'):
                            events_path = os.path.join(meg_dir, file.replace('_meg.fif', '_events.tsv'))
                            if os.path.exists(events_path):
                                events_df = pd.read_csv(events_path, sep='\t')
                                all_labels.extend(events_df['stim_type'].values)
    encoder = LabelEncoder()
    encoder.fit(all_labels)
    return encoder

def data_generator(bids_path, fixed_length=2048, resample_rate=None, transform=None):
    encoder = fit_label_encoder(bids_path)
    print(encoder.classes_)
    file_info_df = load_file_info(bids_path)

    for idx in tqdm(range(len(file_info_df))):
        file_info = file_info_df.iloc[idx]
    
        segments = preprocess_file(file_info, encoder, fixed_length=fixed_length, resample_rate=resample_rate, transform=transform)
        if segments:
            for segment in segments:
                yield segment


seq_len = 512
schema = {'eeg': f'ndarray:float32:74,{seq_len}', 'label':f'ndarray:int32:{seq_len}', 'subject':'str'}
bids_path = '/media/elijah/T7/ds000117-download/'
output_path = '/media/elijah/T7/ds117_streaming_eeg/'
with MDSWriter(out=output_path, columns=schema) as out:
    for data in data_generator(bids_path, fixed_length=seq_len, resample_rate=256):
        # print(data)
        # print(data)
        # if not (data['label']==data['label'][0]).all():
        # print(data)
        out.write(data)
