from time import gmtime, strftime

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy import prod

from config import *
from net import SoundStream, WaveDiscriminator, STFTDiscriminator
from dataset import NSynthDataset
from losses import *
from utils import collate_fn, overall_stft, log_history, save_master_checkpoint, pad_exception, load_master_checkpoint, \
    copy_data_to_node, ActivationStatisticsHook, log_activations

if not os.path.exists(SAVE_FOLDER):
    # If not, create the directory
    os.makedirs(SAVE_FOLDER)

# Need to check this division by 320, because this is a 'compression' parameter made of Strides in encoder/decoder block
# If the length of the audio would not be divisible by 320, then x != G_x
# prod(STRIDES) == 320
assert TENSOR_CUT % prod(STRIDES) == 0
# at least ~1500 length because discriminator kernel 7 by 7 and otherwise would be 6 (hop 256*6 ~ 1500)
assert TENSOR_CUT > 2000
# need to specify at least 2 as a batch, because there is a squeeze in the stft
assert BATCH_SIZE > 1

# HPC data transfer + decompression
train_path_node, test_path_node = copy_data_to_node(TRAIN_FILE, TEST_FILE)

train_dataset = NSynthDataset(audio_dir=train_path_node, sample_rate=SR, tensor_cut=TENSOR_CUT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
print("Train data length: ", len(train_dataset))

test_dataset = NSynthDataset(audio_dir=test_path_node, sample_rate=SR, tensor_cut=TENSOR_CUT)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
print("Test data length: ", len(test_dataset))

# Initialization of the models
# Params: dims = 128 like in Encodec (dim of codes of codebook), codebook_size=1024 like in encodec
# channels=8 ~ 1.5 mil params for model
# n_q = 8 like in Encodec, using 8 times quantization
soundstream = SoundStream(channels=8, dim=128, n_q=8, codebook_size=1024)
wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
# C is the channels in residual blocks of discriminator. The number is not specified in paper
stft_disc = STFTDiscriminator(C=2, F_bins=W // 2)

soundstream.to(DEVICE)
wave_disc.to(DEVICE)
stft_disc.to(DEVICE)

# Loss for discriminator
criterion_d = adversarial_d_loss

optimizer_g = optim.Adam(soundstream.parameters(), lr=LR, betas=(0.5, 0.9))
optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()), lr=LR, betas=(0.5, 0.9))

# If continue training, load states for all models + optimizers
if RESUME:
    load_master_checkpoint(CHECKPOINT_REPOSITORY, CHECKPOINT_NAME,
                           soundstream, optimizer_g, optimizer_d,
                           wave_disc, stft_disc)

best_test_loss = float("inf")

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
        norm = 10
        grad_norm = torch.nn.utils.clip_grad_norm_(soundstream.parameters(), max_norm=float(norm))
        optimizer_g.step()

        history[f"{epoch}"]["loss"].append(loss_g.detach().item())
        history[f"{epoch}"]["grad_norm"].append(grad_norm.detach().item())

        if epoch > N_WARMUP_EPOCHS or RESUME:
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

    with torch.no_grad():
        stft_disc.eval()
        wave_disc.eval()

        test_loss_d = 0.0
        test_loss_g = 0.0
        for x, lengths_x in tqdm(test_loader):
            x = x.to(DEVICE)
            lengths_x = lengths_x.to(DEVICE)

            # Exception
            if x.shape[2] % prod(STRIDES) != 0:
                x, lengths_x = pad_exception(x, TENSOR_CUT, DEVICE)

            G_x = soundstream(x)

            stft_x = overall_stft(x, device=DEVICE)
            stft_G_x = overall_stft(G_x, device=DEVICE)

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

            # Save the model
            if test_loss_g < best_test_loss:
                save_name = os.path.join(SAVE_FOLDER, f"la_{LAMBDA_ADV}_lr_{LAMBDA_REC}_lf_{LAMBDA_FEAT}_ep_{epoch}_" \
                                         + strftime("%m-%d_%H:%M", gmtime()) + ".cpt")
                best_test_loss = test_loss_g
                save_master_checkpoint(soundstream, optimizer_d, optimizer_g, wave_disc, stft_disc, save_name)

        print(f"Epoch {epoch}, test gen loss is {test_loss_g / len(train_loader)}")
        print(f"Epoch {epoch}, test disc loss is {test_loss_d / len(train_loader)}")

    # At the end of epoch rewrite losses dict
    log_history(history, SAVE_FOLDER)

    # Save only means of means of activations and means of std of activations
    hook.aggregate_mean()

    log_activations(activations, hook, SAVE_ACTIVATIONS, epoch)
