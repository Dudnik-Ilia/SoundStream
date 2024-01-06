import os
import torch

# Lambdas for loss weighting
LAMBDA_ADV = 1
LAMBDA_FEAT = 100
LAMBDA_REC = 1
lambdas = [LAMBDA_ADV, LAMBDA_FEAT, LAMBDA_REC]

# Learning params
STRIDES = [8, 5, 4, 2]
TENSOR_CUT = 64000
N_EPOCHS = 15
N_WARMUP_EPOCHS = 3
TRAIN_DISC_EVERY = 2
BATCH_SIZE = 6
LR = 1e-8
SAVE_FOLDER = os.path.join("/home/woody/iwi1/iwi1010h/checkpoints/SoundStream/", os.environ['SLURM_JOBID'])
SAVE_ACTIVATIONS = SAVE_FOLDER

# Windows length and hop for stft
W, H = 1024, 256
SR = 24000

# Data names that should be at WORK
TRAIN_FILE = "train-clean-100.tar.gz"
TEST_FILE = "test-clean.tar.gz"

# If continue training
RESUME = False
CHECKPOINT_REPOSITORY = ""
CHECKPOINT_NAME = ""

DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))