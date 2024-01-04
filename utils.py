import os
import pickle
import shutil
import tarfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from net import EncoderBlock, DecoderBlock


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
    x = x.squeeze(1)  # delete dimensions of 1 (channel)
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
    optimizer_g.load_state_dict(state_dict['scheduler_g_dict'])
    wave_disc.load_state_dict(state_dict['wave_disc_dict'])
    stft_disc.load_state_dict(state_dict['stft_disc_dict'])

    return 0


def copy_data_to_node(train_file, test_file):
    """
    The following is the part for copying the data to the node
    for Cloud computing purposes
    Following env var needed: WORK, TMPDIR, SLURM_JOBID
    Returns: folders with audios (like Librispeech)
    """
    train_file_work_path = os.path.join(os.environ['WORK'], train_file)
    test_file_work_path = os.path.join(os.environ['WORK'], test_file)

    train_path_node = os.path.join(os.environ['TMPDIR'], os.environ['SLURM_JOBID'], "train_data")
    test_path_node = os.path.join(os.environ['TMPDIR'], os.environ['SLURM_JOBID'], "test_data")
    os.makedirs(train_path_node)
    os.makedirs(test_path_node)
    print("Created tmp paths: ", "\n", train_path_node, "\n", test_path_node)

    # COPY THE DATA from WORK to the Node at $TMPDIR/$SLURM_JOBID/Train(Test)
    shutil.copy(train_file_work_path, train_path_node)
    shutil.copy(test_file_work_path, test_path_node)
    print("Copied archives")

    # Tars are in the 'train_path_node' and 'test_path_node'
    # Extract the inhalt to the same paths, so they will be at the same place where Tars are
    file = tarfile.open(os.path.join(train_path_node, train_file))
    file.extractall(train_path_node)
    file.close()
    print("Unzipped train")
    file = tarfile.open(os.path.join(test_path_node, test_file))
    file.extractall(test_path_node)
    file.close()
    print("Unzipped test")

    return train_path_node, test_path_node


class ActivationStatisticsHook:
    def __init__(self, model):
        self.activation_means = {}
        self.activation_stds = {}

        # Register hook for each module in the model
        for name, module in model.named_modules():
            if isinstance(module, EncoderBlock) or isinstance(module, DecoderBlock):
                # Store the custom names specified during initialization
                custom_name = getattr(module, 'custom_name', None)
                if custom_name:
                    self.activation_means[custom_name] = []
                    self.activation_stds[custom_name] = []
                module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        mean = output.mean().item()
        std = output.std().item()

        custom_name = getattr(module, 'custom_name', None)
        if custom_name:
            self.activation_means[custom_name].append(mean)
            self.activation_stds[custom_name].append(std)

    def clear_statistics(self):
        for key in self.activation_means.keys():
            self.activation_means[key] = []
            self.activation_stds[key] = []

    def aggregate_mean(self):
        for key, value in self.activation_means.items():
            self.activation_means[key] = np.mean(value)
        for key, value in self.activation_stds.items():
            self.activation_stds[key] = np.mean(value)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_history(history: dict, save_folder):
    """
    Log history dict with losses
    Rewrites the file if the file exists
    """
    save_name = os.path.normpath(os.path.join(save_folder, f"history.p"))
    pickle.dump(history, open(save_name, "wb"))


def log_activations(activations: dict, hook: ActivationStatisticsHook, save_folder: str, epoch: int):
    """
    Log history dict with losses
    Rewrites the file if the file exists
    """
    activations_cur_ep = {
        "mean": hook.activation_means,
        "std": hook.activation_stds
    }

    activations[str(epoch)] = activations_cur_ep
    save_name = os.path.normpath(os.path.join(save_folder, "activations.p"))
    pickle.dump(activations, open(save_name, "wb"))


"""
# Test case
batch = [torch.rand(size=(1,100)),
     torch.rand(size=(1,200))]
batch, length = pad_exception(batch,320,"cpu")
print(batch.shape)
"""