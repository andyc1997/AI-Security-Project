import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Description: This script contains the encoder-decoder networks designed for MNIST dataset.
'''

# network dimensions
c = 32  # capacity
p_f = 128  # output dimension
alpha = 0.2 # slope for leaky relu

class Encoder(nn.Module):
    """
    Encoder function
    :param int p_feat: The dimension of feature space
    """

    def __init__(self, p_latent: int = p_f):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(c * 2 * 7 * 7, 2 * p_latent)

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(1, 1, 28, 28)
        x = F.leaky_relu(self.conv1(x), negative_slope=alpha)
        x = F.leaky_relu(self.conv2(x), negative_slope=alpha)
        x = x.view(x.size(0), c * 2 * 7 * 7)
        x = self.fc1(x)
        return x  # p_f


class Decoder(nn.Module):
    """
    Decoder function
    :param int p_latent: The dimension of the latent space
    """

    def __init__(self, p_latent: int = p_f):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=p_latent, out_features=c * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=alpha)
        if x.dim() == 1:
            x = x.view(1, c * 2, 7, 7)
        else:
            x = x.view(x.size(0), c * 2, 7, 7)
        x = F.leaky_relu(self.conv2(x), negative_slope=alpha)
        x = torch.sigmoid(self.conv1(x))
        return x  # N x 28 x 28
