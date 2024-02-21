import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib

@st.cache_resource
def list_directories(path):
    """ Returns a list of directories in a given path """
    path = Path(path)
    if path.is_dir():
        return [d.name for d in path.iterdir() if d.is_dir()]
    else:
        st.error(f'Path not found: {path}')
        return []

@st.cache_resource
def get_subject_list(bids_dir):
    """ Returns a sorted list of subjects in the BIDS directory """
    subjects = [d.name for d in Path(bids_dir).rglob('sub-*') if d.is_dir()]
    return sorted(subjects)

@st.cache_resource
def load_mri_data(mri_file):
    """ Load and return MRI data """
    mri_image = nib.load(mri_file)
    return mri_image.get_fdata()

def show_mri_image(data, slice_indices):
    """ Display MRI slices using matplotlib with adjustable slice indices """
    fig, axes = plt.subplots(1, 3)
    for i, axis in enumerate(axes):
        slice = data.take(slice_indices[i], axis=i)
        axis.imshow(slice.T, cmap="gray", origin="lower")
    st.pyplot(fig)

def app():
    st.title('MRI Analyzer')

    base_path = Path.home()

    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = base_path

    st.session_state.dirs = list_directories(st.session_state.current_path)
    selected_dir = st.selectbox("Select a directory:", st.session_state.dirs)

    if st.button("Go to directory"):
        st.session_state.current_path /= selected_dir
        st.session_state.dirs = list_directories(st.session_state.current_path)

    if st.button('Reset path to home'):
        st.session_state.current_path = base_path
        st.session_state.dirs = list_directories(base_path)

    st.write("Current path:", st.session_state.current_path)
    files = list(st.session_state.current_path.glob('*'))
    st.write("Number of files in current directory:", len(files))

    subject_list = get_subject_list(st.session_state.current_path)
    selected_subject = st.selectbox('Select a subject', subject_list, key='subject')

    if selected_subject:
        mri_path = st.session_state.current_path / selected_subject / 'anat'
        mri_files = [f.name for f in mri_path.glob('*.nii*')]

        selected_mri_file = st.selectbox('Select an MRI image', mri_files, key='mri_file')
        if selected_mri_file:
            data = load_mri_data(mri_path / selected_mri_file)
            data_shape = data.shape

            slice_indices = []
            for i in range(3):
                max_index = data_shape[i] - 1
                index = st.slider(f'Slice index for axis {i}', 0, max_index, max_index // 2, key=f'slider_{i}_{selected_subject}_{selected_mri_file}')
                slice_indices.append(index)

            show_mri_image(data, slice_indices)

if __name__ == "__main__":
    app()
