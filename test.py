from net import SoundStream, WaveDiscriminator, STFTDiscriminator, save_master_checkpoint
from losses import *
import torch.optim as optim
import torch

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

LAMBDA_ADV = 1
LAMBDA_FEAT = 100
LAMBDA_REC = 1
lambdas = [LAMBDA_ADV, LAMBDA_FEAT, LAMBDA_REC]

SR = 24000
LR = 1e-4
DEVICE = "cpu"

soundstream = SoundStream(channels=1, dim=32, n_q=4, codebook_size=128)
wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
stft_disc = STFTDiscriminator(C=1, F_bins=1024 // 2)

torch.manual_seed(10)
# need to specify at least 2 as a batch, because there is a squeeze in the stft
# at least ~1500 length because discriminator kernel 7 by 7 and otherwise would be 6 (hop 256*6 ~ 1500)
x = torch.rand((2,1,320*16))
lengths_x = torch.tensor([320*4])
encoded = soundstream(x)

criterion_d = adversarial_d_loss

optimizer_g = optim.Adam(soundstream.parameters(), lr=LR, betas=(0.5, 0.9))
optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))

soundstream.train()
stft_disc.train()
wave_disc.train()

train_loss_d = 0.0
train_loss_g = 0.0

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
                     lengths_wave, lengths_stft, DEVICE, SR, lambdas)
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
