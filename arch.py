import torch as t
import torch.nn as nn


class beta_VAE_chairs(nn.Module):
    def __init__(self, k = 32):
        super(beta_VAE_chairs, self).__init__()

        self.bias_bool = False

        self.latent_dim = k
        self.encoder = nn.Sequential( 
            nn.Conv2d(1,32, kernel_size = (4,4), stride = 2, bias = self.bias_bool), #input = 1x64x64 output = 32x31x31
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=32), 
            nn.Conv2d(32,32, kernel_size = (4,4), stride = 2, bias = self.bias_bool), #output = 32x14x14 (this one layer uses the floor function in the formula for the outputsize, so we need output_padding = 1 in th ereverse conv)
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32,64, kernel_size = (4,4), stride = 2, bias = self.bias_bool), #output = 64x6x6
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64,64, kernel_size = (4,4), stride = 2, bias = self.bias_bool), #output = 64x2x2
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 2*k)
        )


        self.decoder = nn.Sequential(
        nn.Linear(k, 256), #output = 256
        nn.ReLU(),
        nn.Linear(256, 256), #output = 256
        nn.ReLU(),
        nn.Unflatten(dim=1, unflattened_size=(64, 2, 2)), #output = 64x2x2
        nn.ConvTranspose2d(in_channels = 64, out_channels=64, kernel_size=(4,4), stride = 2, bias = self.bias_bool), #output = 64x6x6
        nn.ReLU(),
        nn.BatchNorm2d(num_features=64),
        nn.ConvTranspose2d(in_channels = 64, out_channels=32, kernel_size=(4,4), stride = 2, bias = self.bias_bool), #output = 32x14x14
        nn.ReLU(),
        nn.BatchNorm2d(num_features=32),
        nn.ConvTranspose2d(in_channels = 32, out_channels=32, kernel_size=(4,4), stride = 2, output_padding = 1, bias = self.bias_bool), #output = 32x31x31 
        nn.ReLU(),
        nn.BatchNorm2d(num_features=32),
        nn.ConvTranspose2d(in_channels = 32, out_channels=1, kernel_size=(4,4), stride = 2, bias = self.bias_bool),
        nn.ReLU() #output should be 1x64x64, where each entry is a probability 
        )

        self.encoder_output = None
        self.decoder_output = None

    def reconstruct(self, sample: bool):
        #Beta VAE paper seems to just take the mean. We could do that or sample from cont. Bernoulli
        if sample:
            return t.distributions.continuous_bernoulli.ContinuousBernoulli(probs = self.decoder_output)
        else:
            return self.decoder_output

    def forward(self, input):
        self.encoder_output = self.encoder(input) #Shape B x (2*k)
        mu, log_sigma = t.split(self.encoder_output, self.latent_dim, dim=1)
        sigma = t.exp(log_sigma)
        Sampler = t.distributions.MultivariateNormal(t.zeros(self.latent_dim), t.eye(self.latent_dim))
        epsilon = Sampler.sample()
        # epsilon = t.normal(t.zeros(self.latent_dim), t.eye(self.latent_dim)) #TODO: doublecheck mean dimension
        latent = mu + sigma * epsilon #k-dim vector
        print(f"{mu=}")
        print(f"{epsilon=}")
        print(f"{sigma=}")

        print(latent.shape)
        print(latent)
        self.decoder_output = self.decoder(latent)

        return self.decoder_output


