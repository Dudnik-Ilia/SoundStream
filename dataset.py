import torch
import torchaudio
import os
from torch.utils.data import Dataset
import random

# Accumulate all file names
def get_file_names(start_dir: str):
    """
    Look in the start_dir for all files ending with flac or wav
    Returns lists with full paths and just names (was meant to be used in conversion and saving)
    """
    path_list = []
    file_list = []
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.lower().endswith(('.flac', '.wav')):
                file_list.append(file)
                path_list.append(os.path.join(root, file))
    return path_list, file_list


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    """
    Audio resampling to the target sample rate
    """
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""
    
    def __init__(self, audio_dir, sample_rate=24000, tensor_cut=36000):
        super().__init__()
        self.filenames, _ = get_file_names(audio_dir)
        self.sr = sample_rate
        self.tensor_cut = tensor_cut

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        """
        During getting an item we transform it to needed sample rate
        """
        waveform, sample_rate = torchaudio.load(self.filenames[index])
        if sample_rate != self.sr:
            waveform = convert_audio(wav=waveform, sr=sample_rate, target_sr=self.sr, target_channels=1)

        # Cut the length of audio
        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                # random start point
                start = random.randint(0, waveform.size()[1] - self.tensor_cut - 1)
                # cut tensor
                waveform = waveform[:, start:start + self.tensor_cut]
        return waveform
