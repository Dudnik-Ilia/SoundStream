import os
import pickle
from time import strftime, gmtime
import torch
from torch import nn

def log_history(history: dict, save_folder):
    """
    Log history dict with losses
    Rewrites the file if the file exists
    """
    save_name = os.path.join(save_folder, f"history_"\
                             + strftime("%m-%d_%H:%M", gmtime()) + ".p")
    pickle.dump(history, open(save_name, "wb"))


# make equal length
def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    return nn.utils.rnn.pad_sequence(batch, batch_first=True), lengths


def overall_stft(x: torch.Tensor, window_length=1024, hop=256, device='cpu'):
    """
    stft used everywhere the same
    """
    x = x.squeeze() # delete dimensions of 1 (channel)
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