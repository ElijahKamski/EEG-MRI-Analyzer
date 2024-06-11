import torch

class EEGFourierTransform:
    def __call__(self, eeg_data):
        """
        Apply Fourier Transform to an EEG timeseries with multiple channels using PyTorch's FFT.
        
        Parameters:
        eeg_data (torch.Tensor): The EEG data tensor, expected shape is (channels, timepoints).
        
        Returns:
        torch.Tensor: The Fourier Transform (magnitude) of the EEG data, shape is (channels, frequency_bins).
        """
        # Ensure eeg_data is a PyTorch tensor
        if not isinstance(eeg_data, torch.Tensor):
            raise TypeError("EEG data must be a PyTorch tensor.")

        # Perform the FFT along the timepoints axis
        fft_result = torch.fft.rfft(eeg_data, dim=1)
        
        # Calculate the magnitude of the complex numbers (amplitude spectrum)
        magnitude = torch.abs(fft_result)
        
        return magnitude
    
    
class TransposeTransform:
    def __call__(self, eeg_data):
        return eeg_data.T
