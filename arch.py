import torch as t
import torch.nn as nn

class beta_VAE_chairs(nn.Module):
    def __init__(self, k = 32):
        super(self).__init__()

        self.latent_dim = k
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32, kernel_size = (4,4), stride = 2), #input = 64x64x1 output = 31x31x32
            nn.Conv2d(32,32, kernel_size = (4,4), stride = 2), #output = 14x14x32
            nn.Conv2d(32,64, kernel_size = (4,4), stride = 2), #output = 6x6x64
            nn.Conv2d(64,64, kernel_size = (4,4), stride = 2), #output = 2x2x64
            nn.Flatten(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 2*k)
        )



    def forward(self, input):
        encoder_output = self.encoder(input)
        mu, sigma = t.split(encoder_output, self.latent_dim)
        epsilon = t.normal(t.zeros(self.latent_dim), t.eye(self.latent_dim)) #TODO: doublecheck mean dimension
        latent = mu + epsilon * sigma #k-dim vector
        self.decoder(latent)



