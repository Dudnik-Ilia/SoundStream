import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

def adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()

    stft_loss = F.relu(1 - features_stft_disc_G_x[-1]).sum(dim=3).squeeze() / lengths_stft[-1].squeeze()
    wave_loss = torch.cat(
        [F.relu(1 - features_wave_disc_G_x[key][-1]).sum(dim=2).squeeze() / lengths_wave[key][-1].squeeze() for key in
         wave_disc_names])
    loss = torch.cat([stft_loss, wave_loss]).mean()

    return loss


def feature_loss(features_stft_disc_x, features_wave_disc_x,
                 features_stft_disc_G_x, features_wave_disc_G_x,
                 lengths_wave, lengths_stft):
    wave_disc_names = lengths_wave.keys()

    for layer_x, layer_G_x in zip(features_stft_disc_x, features_stft_disc_G_x):
        assert layer_x.shape == layer_G_x.shape

    stft_loss = torch.stack(
        [((feat_x - feat_G_x).abs().sum(dim=-1) / lengths_stft[i].view(-1, 1, 1)).sum(dim=-1).sum(dim=-1) for
         i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))], dim=1).mean(dim=1,
                                                                                                             keepdim=True)
    wave_loss = torch.stack([torch.stack(
        [(feat_x - feat_G_x).abs().sum(dim=-1).sum(dim=-1) / lengths_wave[key][i] for i, (feat_x, feat_G_x) in
         enumerate(zip(features_wave_disc_x[key], features_wave_disc_G_x[key]))], dim=1) for key in wave_disc_names],
        dim=2).mean(dim=1)
    loss = torch.cat([stft_loss, wave_loss], dim=1).mean()

    return loss

def temporal_reconstruction_loss(x, G_x):
    loss = torch.nn.L1Loss(reduction="sum")
    return loss(x, G_x)


def spectral_reconstruction_loss(x, G_x, eps=1e-5, device="cpu", sr=24000):
    L = 0
    for i in torch.arange(6, 12):
        s = 2 ** i
        alpha_s = (s / 2) ** 0.5
        melspec = MelSpectrogram(sample_rate=sr, n_fft=s, win_length=s,
                                 hop_length=s // 4, n_mels=64).to(device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)

        # Summ the freq mel dim, cause loss vector is applied to freq mel coefs
        # Then take mean over time+batch but summ over samples in the batch
        loss_1 = (S_x - S_G_x).abs().sum(dim=2).mean()
        loss_2 = torch.log(S_x + eps) - torch.log(S_G_x + eps)
        loss_2 = torch.pow(loss_2, 2).sum(dim=2).sqrt().mean()

        L += loss_1 + alpha_s * loss_2

    return L


def adversarial_d_loss(features_stft_disc_x, features_wave_disc_x,
                       features_stft_disc_G_x, features_wave_disc_G_x,
                       lengths_stft, lengths_wave):
    """
    Loss for discriminator: Hinge loss
    """
    wave_disc_names = lengths_wave.keys()

    real_stft_loss = F.relu(1 - features_stft_disc_x[-1]).sum(dim=3).squeeze() / lengths_stft[-1].squeeze()
    real_wave_loss = torch.stack(
        [F.relu(1 - features_wave_disc_x[key][-1]).sum(dim=-1).squeeze() / lengths_wave[key][-1].squeeze() for key in
         wave_disc_names], dim=1)
    real_loss = torch.cat([real_stft_loss.view(-1, 1), real_wave_loss], dim=1).mean()

    generated_stft_loss = F.relu(1 + features_stft_disc_G_x[-1]).sum(dim=-1).squeeze() / lengths_stft[-1].squeeze()
    generated_wave_loss = torch.stack(
        [F.relu(1 + features_wave_disc_G_x[key][-1]).sum(dim=-1).squeeze() / lengths_wave[key][-1].squeeze() for key in
         wave_disc_names], dim=1)
    generated_loss = torch.cat([generated_stft_loss.view(-1, 1), generated_wave_loss], dim=1).mean()

    return real_loss + generated_loss


def criterion_g(x, G_x, features_stft_disc_x,
                features_wave_disc_x, features_stft_disc_G_x,
                features_wave_disc_G_x, lengths_wave, lengths_stft,
                sr, device, lambdas):
    """
    Generator loss weighted trough several losses:
    reconstruction, adversarial and feature
    """
    LAMBDA_ADV, LAMBDA_FEAT, LAMBDA_REC = lambdas[0], lambdas[1], lambdas[2]
    LAMBDA_REC_TIME = 1

    adv_loss = LAMBDA_ADV * adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x,
                                               lengths_stft, lengths_wave)
    feat_loss = LAMBDA_FEAT * feature_loss(features_stft_disc_x, features_wave_disc_x,
                                           features_stft_disc_G_x, features_wave_disc_G_x,
                                           lengths_wave, lengths_stft)
    rec_freq_loss = LAMBDA_REC * spectral_reconstruction_loss(x, G_x, device=device, sr=sr)
    rec_time_loss = temporal_reconstruction_loss(x, G_x)

    total_loss = adv_loss + feat_loss + rec_freq_loss + LAMBDA_REC_TIME * rec_time_loss
    return total_loss