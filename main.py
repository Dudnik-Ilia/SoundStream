import os
from time import gmtime, strftime
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from net import SoundStream, WaveDiscriminator, STFTDiscriminator
from dataset import NSynthDataset
from losses import *

# Lambdas for loss weighting
LAMBDA_ADV = 1
LAMBDA_FEAT = 100
LAMBDA_REC = 1
lambdas = [LAMBDA_ADV, LAMBDA_FEAT, LAMBDA_REC]

# Learning params
N_EPOCHS = 2
BATCH_SIZE = 4
LR = 1e-4
SAVE_FOLDER = "/home/woody/iwi1/iwi1010h/checkpoints/SoundStream/"

# Windows length and hop for stft
W, H = 1024, 256
SR = 24000

# If continue training
RESUME = False
CHECKPOINT_REPOSITORY = ""
CHECKPOINT_NAME = ""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# make equal length
def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    return nn.utils.rnn.pad_sequence(batch, batch_first=True), lengths


def overall_stft(x: torch.Tensor, window_length=W, hop=H, device=DEVICE):
    """
    stft used everywhere the same
    """
    return torch.stft(x.squeeze(), n_fft=1024, hop_length=hop,
                      window=torch.hann_window(window_length=window_length, device=device),
                      return_complex=False).permute(0, 3, 1, 2)


def save_master_checkpoint(core_model, optimizer_d, optimizer_g,
                           wave_disc, stft_disc, ckpt_name):
    """save checkpoint
    Args:
        core_model (nn.Module): model
        optimizer_d (optimizer): optimizer for discriminant
        optimizer_g (optimizer): optimizer for discriminant
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


soundstream = SoundStream(channels=1, dim=1, n_q=1, codebook_size=1)
wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
stft_disc = STFTDiscriminator(C=1, F_bins=W // 2)

soundstream.to(DEVICE)
wave_disc.to(DEVICE)
stft_disc.to(DEVICE)

# Data loaders
train_dataset = NSynthDataset(audio_dir="./data/nsynth-train.jsonwav/nsynth-train/audio")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

valid_dataset = NSynthDataset(audio_dir="./data/nsynth-valid.jsonwav/nsynth-valid/audio")
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

test_dataset = NSynthDataset(audio_dir="./data/nsynth-test.jsonwav/nsynth-test/audio")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Loss for discriminator
criterion_d = adversarial_d_loss

optimizer_g = optim.Adam(soundstream.parameters(), lr=LR, betas=(0.5, 0.9))
optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))

# If continue training, load states for all models + optimizers
if RESUME:
    file = os.path.join(CHECKPOINT_REPOSITORY, CHECKPOINT_NAME)
    state_dict = torch.load(file, map_location='cpu')
    soundstream.load_state_dict(state_dict['model_state_dict'])
    optimizer_d.load_state_dict(state_dict['optimizer_d_dict'])
    optimizer_g.load_state_dict(state_dict['optimizer_g_dict'])
    wave_disc.load_state_dict(state_dict['wave_disc_dict'])
    stft_disc.load_state_dict(state_dict['stft_disc_dict'])

best_val_loss = float("inf")

history = {
    "train": {"d": [], "g": []},
    "valid": {"d": [], "g": []},
    "test": {"d": [], "g": []}
}

# Training
for epoch in range(1, N_EPOCHS + 1):

    soundstream.train()
    stft_disc.train()
    wave_disc.train()

    train_loss_d = 0.0
    train_loss_g = 0.0
    for x, lengths_x in tqdm(train_loader):
        x = x.to(DEVICE)
        lengths_x = lengths_x.to(DEVICE)

        # Generated x (output)
        G_x = soundstream(x)

        # Calculate STFT of both X and Generated_X
        stft_x = overall_stft(x)
        stft_G_x = overall_stft(G_x)

        lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")

        # TODO: What are these lengths?
        lengths_stft = stft_disc.features_lengths(lengths_s_x)
        lengths_wave = wave_disc.features_lengths(lengths_x)

        # Run through Discriminators
        features_stft_disc_x = stft_disc(stft_x)
        features_wave_disc_x = wave_disc(x)

        features_stft_disc_G_x = stft_disc(stft_G_x)
        features_wave_disc_G_x = wave_disc(G_x)

        # Calculate loss for generator
        loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x,
                             features_stft_disc_G_x, features_wave_disc_G_x,
                             lengths_wave, lengths_stft, SR, DEVICE, lambdas)
        train_loss_g += loss_g.item()

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # Run once for through Discriminators (because this time need to propagate just discriminator)
        # --> detach generated_X
        features_stft_disc_x = stft_disc(stft_x)
        features_wave_disc_x = wave_disc(x)

        features_stft_disc_G_x_det = stft_disc(stft_G_x.detach())
        features_wave_disc_G_x_det = wave_disc(G_x.detach())

        # Calculate loss for discriminator
        loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det,
                             features_wave_disc_G_x_det, lengths_stft, lengths_wave)

        train_loss_d += loss_d.item()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

    # Average loss per epoch
    history["train"]["d"].append(train_loss_d / len(train_loader))
    history["train"]["g"].append(train_loss_g / len(train_loader))

    with torch.no_grad():
        stft_disc.eval()
        wave_disc.eval()

        valid_loss_d = 0.0
        valid_loss_g = 0.0
        for x, lengths_x in tqdm(valid_loader):
            x = x.to(DEVICE)
            lengths_x = lengths_x.to(DEVICE)

            G_x = soundstream(x)

            stft_x = overall_stft(x)
            stft_G_x = overall_stft(G_x)

            lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")

            lengths_stft = stft_disc.features_lengths(lengths_s_x)
            lengths_wave = wave_disc.features_lengths(lengths_x)

            features_stft_disc_x = stft_disc(stft_x)
            features_wave_disc_x = wave_disc(x)

            features_stft_disc_G_x = stft_disc(stft_G_x)
            features_wave_disc_G_x = wave_disc(G_x)

            loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x,
                                 features_wave_disc_G_x, lengths_wave, lengths_stft, SR, DEVICE, lambdas)
            valid_loss_g += loss_g.item()

            features_stft_disc_x = stft_disc(stft_x)
            features_wave_disc_x = wave_disc(x)

            features_stft_disc_G_x_det = stft_disc(stft_G_x.detach())
            features_wave_disc_G_x_det = wave_disc(G_x.detach())

            loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det,
                                 features_wave_disc_G_x_det, lengths_stft, lengths_wave)

            valid_loss_d += loss_d.item()

        # Save the model
        if valid_loss_g < best_val_loss:
            save_name = os.path.join(SAVE_FOLDER, f"ladv_{LAMBDA_ADV}_lrec_{LAMBDA_REC}_lfeat_{LAMBDA_FEAT}_" \
                                     + strftime("%m-%d_%H:%M", gmtime()) + ".cpt")
            best_val_loss = valid_loss_g
            save_master_checkpoint(soundstream, optimizer_d, optimizer_g, wave_disc, stft_disc, save_name)

        history["valid"]["d"].append(valid_loss_d / len(valid_loader))
        history["valid"]["g"].append(valid_loss_g / len(valid_loader))

    with torch.no_grad():
        stft_disc.eval()
        wave_disc.eval()

        test_loss_d = 0.0
        test_loss_g = 0.0
        for x, lengths_x in tqdm(test_loader):
            x = x.to(DEVICE)
            lengths_x = lengths_x.to(DEVICE)

            G_x = soundstream(x)

            stft_x = overall_stft(x)
            stft_G_x = overall_stft(G_x)

            lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")

            lengths_stft = stft_disc.features_lengths(lengths_s_x)
            lengths_wave = wave_disc.features_lengths(lengths_x)

            features_stft_disc_x = stft_disc(stft_x)
            features_wave_disc_x = wave_disc(x)

            features_stft_disc_G_x = stft_disc(stft_G_x)
            features_wave_disc_G_x = wave_disc(G_x)

            loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x,
                                 features_wave_disc_G_x, lengths_wave, lengths_stft, SR, DEVICE, lambdas)
            test_loss_g += loss_g.item()

            features_stft_disc_x = stft_disc(stft_x)
            features_wave_disc_x = wave_disc(x)

            features_stft_disc_G_x_det = stft_disc(stft_G_x.detach())
            features_wave_disc_G_x_det = wave_disc(G_x.detach())

            loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det,
                                 features_wave_disc_G_x_det, lengths_stft, lengths_wave)

            test_loss_d += loss_d.item()

        history["test"]["d"].append(test_loss_d / len(test_loader))
        history["test"]["g"].append(test_loss_g / len(test_loader))
