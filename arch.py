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
        nn.ConvTranspose2d(in_channels = 32, out_channels=1, kernel_size=(4,4), stride = 2, bias = self.bias_bool), #output should be 1x64x64, where each entry is a logit 
        nn.BatchNorm2d(num_features=1)
        )

        self.encoder_output = None
        self.decoder_output = None
        self.mu = None
        self.sigma = None

    def reconstruct(self):
        #Beta VAE paper seems to just take the mean. We could do that or sample from cont. Bernoulli
        # if sample:
        #     sampler = t.distributions.continuous_bernoulli.ContinuousBernoulli(probs = t.sigmoid(self.decoder_output))
        #     return sampler.sample()
        # else:
        #     return t.sigmoid(self.decoder_output)

        decoder_probs = t.sigmoid(self.decoder_output)

        decoder_output_scaled = t.where(t.abs(decoder_probs - 1.) < 10e-8, 1-10e-8, decoder_probs)
        decoder_output_scaled = t.where(t.abs(decoder_probs) < 10e-8, 10e-8, decoder_output_scaled)

        lam = decoder_output_scaled


        mean = t.where(t.abs(lam - 0.5) < 10e-3, 0.5, lam/(2*lam - 1) + 1/(2*t.atanh(1-2*lam)) )
        return mean

    def forward(self, input):

        self.encoder_output = self.encoder(input) #Shape B x (2*k)
        self.mu, log_sigma = t.split(self.encoder_output, self.latent_dim, dim=1)
        self.sigma = t.exp(log_sigma)
        Sampler = t.distributions.MultivariateNormal(t.zeros(self.latent_dim), t.eye(self.latent_dim))
        epsilon = Sampler.sample()
        # epsilon = t.normal(t.zeros(self.latent_dim), t.eye(self.latent_dim)) #TODO: doublecheck mean dimension
        latent = self.mu + self.sigma * epsilon #k-dim vector
        # print(f"{self.mu=}")
        # print(f"{epsilon=}")
        # print(f"{self.sigma=}")

        # print(f"{self.encoder_output.min()=}")
        # print(f"{self.encoder_output.max()=}")
        # print(f"{self.encoder_output.mean()=}")

        # print(latent.shape)
        # print(latent)
        self.decoder_output = self.decoder(latent)

        return self.decoder_output


