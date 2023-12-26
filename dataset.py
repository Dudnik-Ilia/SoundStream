from torch.utils.data import Dataset
import torchaudio

import glob

class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""
    
    def __init__(self, audio_dir, sample_rate=24000):
        super().__init__()
        
        self.filenames = glob.glob(audio_dir+"/*.wav")
        self.sr = sample_rate

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        return torchaudio.load(self.filenames[index])[0]
