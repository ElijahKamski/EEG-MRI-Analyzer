import os
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import LabelEncoder
import nibabel as nib
from nilearn import image
from streaming import MDSWriter
import argparse

# Define the schema for fMRI data
schema = {
    'func': 'ndarray:float32',  # assuming the shape of fMRI data
    'label': 'ndarray:int32',
    'subject': 'str',
    # 'anat': 'ndarray:float32',
    'affine': 'ndarray:float32'
}

def load_file_info(bids_path):
    file_data = []
    for subject_dir in os.listdir(bids_path):
        full_subject_dir = os.path.join(bids_path, subject_dir)
        if subject_dir.startswith('sub-'):
            for session_dir in os.listdir(full_subject_dir):
                if session_dir.startswith('ses-mri'):
                    func_dir = os.path.join(full_subject_dir, session_dir, 'func')
                    # anat_dir = os.path.join(full_subject_dir, session_dir, 'anat')
                    for file in os.listdir(func_dir):
                        if file.endswith('_bold.nii.gz'):
                            func_path = os.path.join(func_dir, file)
                            events_path = func_path.replace('_bold.nii.gz', '_events.tsv')
                            # anat_path = os.path.join(anat_dir, f"{subject_dir}_{session_dir}_T1w.nii.gz")
                            if os.path.exists(events_path):
                                file_data.append({
                                    'subject': subject_dir.split('-')[-1],
                                    'func_path': func_path,
                                    'events_path': events_path,
                                    # 'anat_path': anat_path
                                })
    return pd.DataFrame(file_data)

def load_subject_data(events_path, encoder):
    events_df = pd.read_csv(events_path, sep='\t')
    session_data = []
    for idx, event in events_df.iterrows():
        session_data.append({
            'start': event['onset'],
            'duration': event['duration'] if 'duration' in event else 1.0,
            'label': event['stim_type'],
        })
    df = pd.DataFrame(session_data)
    df['label'] = encoder.transform(df['label'])
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

def preprocess_file(file_info, encoder, fixed_length=2048, transform=None):
    func_img = nib.load(file_info['func_path'])
    # anat_img = nib.load(file_info['anat_path'])
    affine = func_img.affine
    tr = func_img.header.get_zooms()[3]
    sfreq = 1.0 / tr
    new_len = int(fixed_length * sfreq)
    
    print(f"freq = {sfreq}\t tr = {tr}")
    func_data = func_img.get_fdata()
    total_samples = func_data.shape[3]
    subject = file_info['subject']

    events = load_subject_data(file_info['events_path'], encoder).to_dict('records')

    for start in range(0, total_samples, new_len):
        end = min(start + new_len, total_samples)  # Ensure the last segment is not skipped

        func_segment = func_data[..., start:end]
        if transform:
            func_segment = transform(func_segment)

        segment_length = end - start
        label_vector = create_label_vector(events, sfreq, start, segment_length)
        
        yield {
            'func': func_segment.astype(np.float32),
            'label': label_vector.astype(np.int32),
            'subject': subject,
            # 'anat': anat_img.get_fdata().astype(np.float32),
            'affine': affine.astype(np.float32),
            'new_len': new_len,
        }

def fit_label_encoder(bids_path):
    all_labels = []
    for subject_dir in os.listdir(bids_path):
        full_subject_dir = os.path.join(bids_path, subject_dir)
        if subject_dir.startswith('sub-'):
            for session_dir in os.listdir(full_subject_dir):
                if session_dir.startswith('ses-mri'):
                    func_dir = os.path.join(full_subject_dir, session_dir, 'func')
                    for file in os.listdir(func_dir):
                        if file.endswith('_events.tsv'):
                            events_path = os.path.join(func_dir, file)
                            if os.path.exists(events_path):
                                events_df = pd.read_csv(events_path, sep='\t')
                                all_labels.extend(events_df['stim_type'].values)
    encoder = LabelEncoder()
    encoder.fit(all_labels)
    return encoder

def data_generator(bids_path, fixed_length=2048, transform=None):
    encoder = fit_label_encoder(bids_path)
    print(encoder.classes_)
    file_info_df = load_file_info(bids_path)
    for idx in tqdm(range(len(file_info_df))):
        file_info = file_info_df.iloc[idx]
        segments = preprocess_file(file_info, encoder, fixed_length=fixed_length, transform=transform)
        if segments:
            for segment in segments:
                yield segment

def pad_sequence(sequences, max_len, pad_value=0):
    padded_sequences = np.full((len(sequences), *sequences[0].shape[:-1], max_len), pad_value, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        length = seq.shape[-1]
        padded_sequences[i, ..., :length] = seq
    return padded_sequences

def collate_fn(batch, seq_len=2048):
    func = pad_sequence([batch['func']], seq_len, pad_value=-1)[0]
    label = pad_sequence([batch['label']], seq_len, pad_value=-1)[0]
    subject = batch['subject']
    affine = batch['affine']

    return {
        'func': func,
        'label': label,
        'subject': subject,
        'affine': affine
    }

def main(args):
    bids_path = args['bids_path']
    output_path = args['output_path']
    fixed_length = args['fixed_length']
    transform = None  # Placeholder for any transformation function if needed

    with MDSWriter(out=output_path, columns=schema) as out:
        for data in data_generator(bids_path, fixed_length=fixed_length, transform=transform):
            batched_data = collate_fn(data, seq_len=data['new_len'])
            out.write(batched_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process fMRI data from BIDS format and write to MDS format.')
    parser.add_argument('--bids_path', type=str, default='/media/elijah/T7/ds000117-download/', required=False, help='Path to the BIDS dataset.')
    parser.add_argument('--output_path', type=str, default='/media/elijah/T7/ds000117-mri-streaming/', required=False, help='Path to the output directory.')
    parser.add_argument('--fixed_length', type=int, default=2048, help='Fixed length for data segments.')

    args = parser.parse_args()
    main(args)
# --bids_path /media/elijah/T7/ds000117-download/ --output_path /media/elijah/T7/ds000117-mri-streaming/