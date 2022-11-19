import xxlimited
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pickle
from collections import OrderedDict
from torchvision import transforms


class SingleHidden(nn.Module):
    """
    Example: https://blog.paperspace.com/autoencoder-image-compression-keras/

    Default to MNIST output dimension.
    """

    def __init__(self, input_shape, hidden_dim=300, n_output=784):
        assert isinstance(hidden_dim, int)
        self.hidden_dim = hidden_dim
        super(SingleHidden, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(int(np.prod(input_shape)), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        # sigmoid for scaling from 0 to 1
        x = F.sigmoid(self.linear2(x))
        return x

    def name(self):
        return f"SingleHidden{self.hidden_dim}"


class Conv3(nn.Module):
    """
    https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac#63b2

    TODO : param n_conv layers and just divide by 2 each time, use n_output
    """

    def __init__(self, input_shape, hidden_dim, n_output):
        super(Conv3, self).__init__()

        self.hidden_dim = hidden_dim
        if hidden_dim:
            # else resize to 27x22

            self.flatten = nn.Flatten()

            self.decoder_lin = nn.Sequential(
                nn.Linear(int(np.prod(input_shape)), hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, 27 * 22 * 32),
                nn.ReLU(True),
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 27, 22))

        # https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
        # "Typically, the stride of a convolutional layer is (1×1), that is a filter is moved along one pixel horizontally for each read from left-to-right,"
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=0, output_padding=0
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)

        # sigmoid for scaling from 0 to 1
        x = torch.sigmoid(x)
        return x

    def name(self):
        return f"Conv3_{self.hidden_dim}"


class Conv(nn.Module):
    """
    https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac#63b2

    TODO : param n_conv layers and just divide by 2 each time, use n_output
    """

    def __init__(self, input_shape, hidden_dim, n_output):
        super(Conv, self).__init__()

        self.hidden_dim = hidden_dim
        self.flatten = nn.Flatten()

        self.decoder_lin = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 13 * 11 * 64),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 13, 11))

        # https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
        # "Typically, the stride of a convolutional layer is (1×1), that is a filter is moved along one pixel horizontally for each read from left-to-right,"
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=[0, 1],
                output_padding=[0, 1],
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=0, output_padding=0
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)

        # sigmoid for scaling from 0 to 1
        x = torch.sigmoid(x)
        return x

    def name(self):
        return f"Conv4_{self.hidden_dim}"


class FC2PretrainedStyleGAN(nn.Module):
    """
    Using pre-trained Style GAN models from: https://github.com/NVlabs/stylegan2-ada-pytorch#data-repository

    """

    def __init__(
        self, input_shape, hidden, fp, output_dim=None, grayscale=False, label=None, freeze_gan=True
    ):
        """
        input_shape : Ny x Nx
        hidden : array_like
            Number of nodes per hidden layer. Length of list determines number of hidden layers

        """
        assert hidden is not None
        super(FC2PretrainedStyleGAN, self).__init__()

        # load pretrained model
        with open(fp, "rb") as f:
            G = pickle.load(f)["G_ema"].cuda()  # torch.nn.Module
        generator_latent_dim = G.z_dim
        self.generator = G
        self.label = label  # TODO diff for CIFAR10, i.e. conditional generators
        if freeze_gan:
            self.freeze_gan()
        else:
            self.unfreeze_gan()

        # fully connected to pre-trained GAN
        self.flatten = nn.Flatten()
        layer_dim = [int(np.prod(input_shape))] + hidden
        self.n_layers = len(layer_dim)
        layers = []
        for i in range(len(hidden)):
            layers += [
                (f"linear{i+1}", nn.Linear(layer_dim[i], layer_dim[i + 1], bias=False)),
                (f"bn{i+1}", nn.BatchNorm1d(layer_dim[i + 1])),
                (f"relu{i+1}", nn.ReLU()),
            ]
        layers += [(f"linear{len(layer_dim)}", nn.Linear(layer_dim[-1], generator_latent_dim))]
        self.layers = nn.Sequential(OrderedDict(layers))

        # post-processing
        if output_dim is not None:
            self.resize = transforms.Resize(size=output_dim)
        else:
            self.resize = None
        if grayscale:
            self.grayscale = transforms.Grayscale(num_output_channels=1)
        else:
            self.grayscale = None

    def freeze_gan(self):
        for _, params in self.generator.named_parameters():
            params.requires_grad = False
        self.generator.eval()

    def unfreeze_gan(self):
        for _, params in self.generator.named_parameters():
            params.requires_grad = True
        self.generator.train()

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        x = self.generator(x, self.label)

        # rescale from [-1, 1] to [0, 1]
        x = (x * 0.5 + 0.5).clamp(0, 1)

        if self.grayscale is not None:
            x = self.grayscale(x)
        if self.resize is not None:
            x = self.resize(x)

        return x

    def name(self):
        return f"FC2PretrainedStyleGAN_{self.n_layers}"
