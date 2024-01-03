import os
import pickle
from time import strftime, gmtime
import torch
from torch import nn
import torch.nn.functional as F

def log_history(history: dict, save_folder):
    """
    Log history dict with losses
    Rewrites the file if the file exists
    """
    save_name = os.path.join(save_folder, f"history_" \
                             + strftime("%m-%d_%H:%M", gmtime()) + ".p")
    pickle.dump(history, open(save_name, "wb"))


# make equal length
def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    # Make all tensor in a batch the same length by padding with zeros, padding in function work along first dim
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch, lengths

def pad_exception(x, tensor_cut, device):
    # In case length of padded batch (in time dim) is not div by 320, need to pad it to TENSOR_CUT
    sequences = [t for t in x]
    print(f"Length exception, length of a batch are {[seq.shape[1] for seq in sequences]}")
    padded_sequences = [F.pad(seq, (0, tensor_cut - seq.shape[1])) for seq in sequences]
    x, lengths_x = collate_fn(padded_sequences)
    x = x.to(device)
    lengths_x = lengths_x.to(device)
    return x, lengths_x


def overall_stft(x: torch.Tensor, window_length=1024, hop=256, device='cpu'):
    """
    stft used everywhere the same
    """
    x = x.squeeze()  # delete dimensions of 1 (channel)
    stft = torch.stft(x, n_fft=1024, hop_length=hop,
                      window=torch.hann_window(window_length=window_length, device=device),
                      return_complex=False)
    # Permute to [Batch, Real/Img, Freq, Time]
    stft = stft.permute(0, 3, 1, 2)
    return stft


def save_master_checkpoint(core_model, optimizer_d, optimizer_g,
                           wave_disc, stft_disc, ckpt_name):
    """save checkpoint
    Args:
        core_model (nn.Module): model
        optimizer_d (optimizer): optimizer for discriminant
        optimizer_g (optimizer): optimizer for discriminant
        wave_disc: discriminator of the wave itself
        stft_disc: discriminator of the stft of the wave
        ckpt_name: checkpoint path and name
    """
    state_dict = {
        'model_state_dict': core_model.state_dict(),
        'optimizer_d_dict': optimizer_d.state_dict(),
        'scheduler_g_dict': optimizer_g.state_dict(),
        'wave_disc_dict': wave_disc.state_dict(),
        'stft_disc_dict': stft_disc.state_dict(),
    }
    torch.save(state_dict, ckpt_name)

def load_master_checkpoint(checkpoint_repository, checkpoint_name,
                           soundstream, optimizer_g, optimizer_d,
                           wave_disc, stft_disc):
    """load checkpoint
    Args:
        soundstream (nn.Module): model
        optimizer_d (optimizer): optimizer for discriminant
        optimizer_g (optimizer): optimizer for discriminant
        wave_disc: discriminator of the wave itself
        stft_disc: discriminator of the stft of the wave
        checkpoint_name: checkpoint path and name
        checkpoint_repository: folder
    """

    file = os.path.join(checkpoint_repository, checkpoint_name)
    state_dict = torch.load(file, map_location='cpu')
    soundstream.load_state_dict(state_dict['model_state_dict'])
    optimizer_d.load_state_dict(state_dict['optimizer_d_dict'])
    optimizer_g.load_state_dict(state_dict['optimizer_g_dict'])
    wave_disc.load_state_dict(state_dict['wave_disc_dict'])
    stft_disc.load_state_dict(state_dict['stft_disc_dict'])

    return 0

"""
# Test case
batch = [torch.rand(size=(1,100)),
     torch.rand(size=(1,200))]
batch, length = pad_exception(batch,320,"cpu")
print(batch.shape)
"""