import os.path

from numpy import prod
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

from config_test import *
from dataset import NSynthDataset
from net import SoundStream, WaveDiscriminator, STFTDiscriminator
from losses import *
from utils import collate_fn, overall_stft, pad_exception, log_history, ActivationStatisticsHook, log_activations, \
    count_parameters

train_dataset = NSynthDataset(audio_dir=train_path_node, sample_rate=SR, tensor_cut=TENSOR_CUT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
print("Train data length: ", len(train_dataset))

soundstream = SoundStream(channels=1, dim=32, n_q=4, codebook_size=128)
wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
stft_disc = STFTDiscriminator(C=1, F_bins=1024 // 2)

criterion_d = adversarial_d_loss
optimizer_g = optim.Adam(soundstream.parameters(), lr=LR, betas=(0.5, 0.9))
optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))

hook = ActivationStatisticsHook(soundstream)

history = {}
activations = {}

# Training
for epoch in range(1, N_EPOCHS + 1):

    hook.clear_statistics()

    soundstream.train()
    stft_disc.train()
    wave_disc.train()

    train_loss_d = 0.0
    train_loss_g = 0.0

    # For counting batches
    i = 0

    history[f"{epoch}"] = {"grad_norm": [],
                           "loss": []}

    for x, lengths_x in tqdm(train_loader):
        i += 1

        x = x.to(DEVICE)
        lengths_x = lengths_x.to(DEVICE)

        # Exception, if length not div by 320
        if x.shape[2] % prod(STRIDES) != 0:
            x, lengths_x = pad_exception(x, TENSOR_CUT, DEVICE)

        # Generated x (output)
        G_x = soundstream(x)

        # Calculate STFT of both X and Generated_X
        stft_x = overall_stft(x, device=DEVICE)
        stft_G_x = overall_stft(G_x, device=DEVICE)

        lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")

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

        # for history
        train_loss_g += loss_g.item()

        optimizer_g.zero_grad()
        loss_g.backward()

        # for logging + clipping of the grad
        if loss_g.item() > 1e+6:
            norm = 1000
        elif loss_g.item() > 1e+4:
            norm = 100
        else:
            norm = 10

        grad_norm = torch.nn.utils.clip_grad_norm_(soundstream.parameters(), max_norm=float(norm))
        optimizer_g.step()

        history[f"{epoch}"]["loss"].append(loss_g.detach().item())
        history[f"{epoch}"]["grad_norm"].append(grad_norm.detach().item())

        if epoch > N_WARMUP_EPOCHS:
            if i % TRAIN_DISC_EVERY == 0:
                i = 0
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

    print(f"Epoch {epoch}, train gen loss is {train_loss_g / len(train_loader)}")
    print(f"Epoch {epoch}, train disc loss is {train_loss_d / len(train_loader)}")

    # At the end of epoch rewrite losses dict
    log_history(history, SAVE_FOLDER)

    # Save only means of means of activations and means of std of activations
    hook.aggregate_mean()

    log_activations(activations, hook, SAVE_ACTIVATIONS, epoch)
