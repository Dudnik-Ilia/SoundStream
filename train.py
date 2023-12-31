import os
import shutil
import tarfile
from time import gmtime, strftime
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy import prod

from net import SoundStream, WaveDiscriminator, STFTDiscriminator
from dataset import NSynthDataset
from losses import *
from utils import collate_fn, overall_stft, log_history, save_master_checkpoint, pad_exception

# Lambdas for loss weighting
LAMBDA_ADV = 1
LAMBDA_FEAT = 100
LAMBDA_REC = 1
lambdas = [LAMBDA_ADV, LAMBDA_FEAT, LAMBDA_REC]

# Learning params
STRIDES = [8, 5, 4, 2]
TENSOR_CUT = 64000
N_EPOCHS = 15
BATCH_SIZE = 6
LR = 1e-4
SAVE_FOLDER = "/home/woody/iwi1/iwi1010h/checkpoints/SoundStream/"

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

# Windows length and hop for stft
W, H = 1024, 256
SR = 24000

# Data names that should be at WORK
TRAIN_FILE = "train-clean-360.tar.gz"
TEST_FILE = "test-clean.tar.gz"

# If continue training
RESUME = False
CHECKPOINT_REPOSITORY = ""
CHECKPOINT_NAME = ""

DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# The following is the part for copying the dat to the node
# for Cloud computing purposes
train_file_work_path = os.path.join(os.environ['WORK'], TRAIN_FILE)
test_file_work_path = os.path.join(os.environ['WORK'], TEST_FILE)

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
file = tarfile.open(os.path.join(train_path_node, TRAIN_FILE))
file.extractall(train_path_node)
file.close()
print("Unzipped train")
file = tarfile.open(os.path.join(test_path_node, TEST_FILE))
file.extractall(test_path_node)
file.close()
print("Unzipped test")

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

best_test_loss = float("inf")

history = {
    "train": {"d": [], "g": []},
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

        # Exception, if length not div by 320
        if x.shape[2] % prod(STRIDES) != 0:
            x, lengths_x = pad_exception(x, TENSOR_CUT, DEVICE)

        # Generated x (output)
        G_x = soundstream(x)

        # Calculate STFT of both X and Generated_X
        stft_x = overall_stft(x, device=DEVICE)
        stft_G_x = overall_stft(G_x, device=DEVICE)

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

    print(f"Epoch {epoch}, train gen loss is {train_loss_g / len(train_loader)}")
    print(f"Epoch {epoch}, train disc loss is {train_loss_d / len(train_loader)}")

    # Average loss per epoch
    history["train"]["d"].append(train_loss_d / len(train_loader))
    history["train"]["g"].append(train_loss_g / len(train_loader))

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

        history["test"]["d"].append(test_loss_d / len(test_loader))
        history["test"]["g"].append(test_loss_g / len(test_loader))

    # At the end of epoch rewrite losses dict
    log_history(history, SAVE_FOLDER)
