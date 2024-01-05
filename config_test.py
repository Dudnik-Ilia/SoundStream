import os

LAMBDA_ADV = 1
LAMBDA_FEAT = 100
LAMBDA_REC = 1
lambdas = [LAMBDA_ADV, LAMBDA_FEAT, LAMBDA_REC]
STRIDES = [8, 5, 4, 2]
SR = 24000
LR = 1e-4
DEVICE = "cpu"

# Learning params
N_EPOCHS = 3
BATCH_SIZE = 4
TENSOR_CUT = 32000
N_WARMUP_EPOCHS = 3
TRAIN_DISC_EVERY = 2
train_path_node = str(os.path.normpath("C:\\Study\\Thesis\\Test_compression\\dev-clean\\LibriSpeech\\cut_dev_clean\\"))
SAVE_FOLDER = str(os.path.normpath("C:\\Users\\dudni\\PycharmProjects\\SoundStream\\history\\"))
SAVE_ACTIVATIONS = 'C:/Users/dudni/PycharmProjects/SoundStream/activations'

