import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import os
from vector_quantize_pytorch import ResidualVQ


class CausalConv1d(nn.Conv1d):
    def __init__(self, custom_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.custom_name = custom_name
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x,
                                        [self.causal_padding, 0]),
                                  self.weight,
                                  self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, custom_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.custom_name = custom_name
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[..., :-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            weight_norm(CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=7, dilation=dilation)),
            nn.ELU(),
            weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1))
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride, custom_name=None):
        super().__init__()

        self.custom_name = custom_name
        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels // 2,
                         out_channels=out_channels // 2, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels // 2,
                         out_channels=out_channels // 2, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels // 2,
                         out_channels=out_channels // 2, dilation=9),
            nn.ELU(),
            weight_norm(CausalConv1d(in_channels=out_channels // 2, out_channels=out_channels,
                                     kernel_size=2 * stride, stride=stride))
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride, custom_name=None):
        super().__init__()

        self.custom_name = custom_name
        self.layers = nn.Sequential(
            weight_norm(CausalConvTranspose1d(in_channels=2 * out_channels,
                                              out_channels=out_channels,
                                              kernel_size=2 * stride, stride=stride)),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.layers = nn.Sequential(
            weight_norm(CausalConv1d(in_channels=1, out_channels=C, kernel_size=7)),
            nn.ELU(),
            EncoderBlock(out_channels=2 * C, stride=2, custom_name="encodec.layer.1"),
            nn.ELU(),
            EncoderBlock(out_channels=4 * C, stride=4, custom_name="encodec.layer.2"),
            nn.ELU(),
            EncoderBlock(out_channels=8 * C, stride=5, custom_name="encodec.layer.3"),
            nn.ELU(),
            EncoderBlock(out_channels=16 * C, stride=8, custom_name="encodec.layer.4"),
            nn.ELU(),
            weight_norm(CausalConv1d(in_channels=16 * C, out_channels=D, kernel_size=3, custom_name="encodec.layer.5"))
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.layers = nn.Sequential(
            weight_norm(CausalConv1d(in_channels=D, out_channels=16 * C, kernel_size=7)),
            nn.ELU(),
            DecoderBlock(out_channels=8 * C, stride=8, custom_name="decodec.layer.1"),
            nn.ELU(),
            DecoderBlock(out_channels=4 * C, stride=5, custom_name="decodec.layer.2"),
            nn.ELU(),
            DecoderBlock(out_channels=2 * C, stride=4, custom_name="decodec.layer.3"),
            nn.ELU(),
            DecoderBlock(out_channels=C, stride=2, custom_name="decodec.layer.4"),
            nn.ELU(),
            weight_norm(CausalConv1d(in_channels=C, out_channels=1, kernel_size=7, custom_name="decodec.layer.5"))
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SoundStream(nn.Module):
    def __init__(self, channels, dim, n_q, codebook_size):
        super().__init__()
        # D in Encoder stands for Output_channel
        self.encoder = Encoder(C=channels, D=dim)
        # dim is the dim of output of encode (input of decoder)
        # inside ResidualVQ also codebook_dim which is the dim of codes
        # if not set (like here), then dim is the codes dim too
        self.quantizer = ResidualVQ(
            num_quantizers=n_q, dim=dim, codebook_size=codebook_size,
            shared_codebook=True, quantize_dropout=True,
            kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=channels, D=dim)

    def forward(self, x):
        e = self.encoder(x)
        # Very important: output of encoder is [B, C, T], where C is channels, which meant to be dim in the args
        # dim should be also equal to the code length in codebook
        e = e.permute(0, 2, 1)
        # quantizer waits for the input like [B, T, dim]
        quantized, _, _ = self.quantizer(e)
        # Need to swap back because decoder waits for [B, C, T]
        quantized = quantized.permute(0, 2, 1)
        o = self.decoder(quantized)
        return o

    def load_pretrained(self, checkpoint_name: str, repository: str):
        """
        Should be used for inference purpose
        """
        file = os.path.join(repository, checkpoint_name)
        state_dict = torch.load(file, map_location='cpu')
        self.load_state_dict(state_dict['model_state_dict'])


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


# Wave-based Discriminator
class WaveDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                WNConv1d(in_channels=1, out_channels=16, kernel_size=15),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=16, out_channels=64, kernel_size=41,
                         stride=4, padding=20, groups=4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=64, out_channels=256, kernel_size=41,
                         stride=4, padding=20, groups=16),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=256, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=64),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=256),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=5,
                         stride=1, padding=2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            WNConv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1,
                     padding=1)
        ])

    def features_lengths(self, lengths):
        return [
            lengths,
            torch.div(lengths + 3, 4, rounding_mode="floor"),
            torch.div(lengths + 15, 16, rounding_mode="floor"),
            torch.div(lengths + 63, 64, rounding_mode="floor"),
            torch.div(lengths + 255, 256, rounding_mode="floor"),
            torch.div(lengths + 255, 256, rounding_mode="floor"),
            torch.div(lengths + 255, 256, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map


class WaveDiscriminator(nn.Module):
    def __init__(self, num_D, downsampling_factor):
        super().__init__()

        self.num_D = num_D
        self.downsampling_factor = downsampling_factor

        self.model = nn.ModuleDict({
            f"disc_{downsampling_factor ** i}": WaveDiscriminatorBlock()
            for i in range(num_D)
        })
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=1,
                                        count_include_pad=False)

    def features_lengths(self, lengths):
        return {
            f"disc_{self.downsampling_factor ** i}": self.model[
                f"disc_{self.downsampling_factor ** i}"].features_lengths(
                torch.div(lengths, 2 ** i, rounding_mode="floor")) for i in range(self.num_D)
        }

    def forward(self, x):
        results = {}
        for i in range(self.num_D):
            disc = self.model[f"disc_{self.downsampling_factor ** i}"]
            results[f"disc_{self.downsampling_factor ** i}"] = disc(x)
            x = self.downsampler(x)
        return results


# STFT-based Discriminator

class ResidualUnit2d(nn.Module):
    def __init__(self, in_channels, N, m, s_t, s_f):
        super().__init__()

        self.s_t = s_t
        self.s_f = s_f

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=N,
                kernel_size=(3, 3),
                padding="same"
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=N,
                out_channels=m * N,
                kernel_size=(s_f + 2, s_t + 2),
                stride=(s_f, s_t)
            )
        )

        self.skip_connection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=m * N,
            kernel_size=(1, 1), stride=(s_f, s_t)
        )

    def forward(self, x):
        return self.layers(F.pad(x, [self.s_t + 1, 0, self.s_f + 1, 0])) + self.skip_connection(x)


class STFTDiscriminator(nn.Module):
    """
    Forward method returns feature maps
    """

    def __init__(self, C, F_bins):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7, 7)),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=32, N=C, m=2, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=2 * C, N=2 * C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4 * C, N=4 * C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4 * C, N=4 * C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8 * C, N=8 * C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8 * C, N=8 * C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Conv2d(in_channels=16 * C, out_channels=1,
                      kernel_size=(F_bins // 2 ** 6, 1))
        ])

    def features_lengths(self, lengths):
        return [
            lengths - 6,
            lengths - 6,
            torch.div(lengths - 5, 2, rounding_mode="floor"),
            torch.div(lengths - 5, 2, rounding_mode="floor"),
            torch.div(lengths - 3, 4, rounding_mode="floor"),
            torch.div(lengths - 3, 4, rounding_mode="floor"),
            torch.div(lengths + 1, 8, rounding_mode="floor"),
            torch.div(lengths + 1, 8, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map
