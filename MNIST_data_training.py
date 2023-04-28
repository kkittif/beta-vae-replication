#%%
import torch as t 
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops import rearrange, reduce, repeat
import math 

# Import MNIST data and split into batches
MNIST_data = datasets.MNIST(
    root='data',
    train = True,
    transform = Compose([ToTensor(), Resize((64, 64)),]),
    download= True)

training_dataloader = DataLoader(MNIST_data, batch_size=50, shuffle = True)

#%% 
from arch import beta_VAE_chairs
beta_VAE_MNIST = beta_VAE_chairs(k = 10)

#%%
#Testing a forward pass
image1 = MNIST_data[0][0]
image2 = MNIST_data[1][0]

images = t.stack((image1, image2), dim=0)
bernoulli_means = beta_VAE_MNIST(images)

# plt.imshow(bernoulli_means[0].detach().reshape(64, 64, 1))

#%%

#Training the  net on small amount of data
#config = {'beta' : 1 }

beta = 1
def train_one_epoch(model, dataloader, loss_fun) -> float:
    optimizer = t.optim.Adam(model.parameters())
    total_loss = 0
    count = 0 
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        decoder_output = model(batch_x)
        loss = loss_fun(model, batch_x, decoder_output, model.encoder_output, beta)
        print(f"{loss=}")
        total_loss += loss.item() * len(batch_x)
        loss.backward()
        nan_count = 0 
        for params in model.parameters():
            if math.isnan(params.grad.mean().item()):
                nan_count += 1
        # print(f"{nan_count=}")
        optimizer.step()
        count += 1
        if count > 10:
            break
        
    return (total_loss / len(dataloader.dataset))   #.item()

def loss_bernoulli(model, input, decoder_output, encoder_output, beta) -> float:

    C = t.where(t.abs(decoder_output - 0.5) > 0.01, 2*t.atanh(1-2*0.99*decoder_output)/(1-2*0.99*decoder_output), 2.0) 
    
    # TODO: What is the decoder output is really close to 0.5? Taylor-expansion
    reconstruction_loss = -reduce((t.log(C) + input*t.log(decoder_output) + (1 - input) * t.log(1 - decoder_output)), 'b c h w -> b ', reduction = 'sum') # shape: batch

    mu_squared = t.einsum('...i,...i -> ...', [model.mu, model.mu])

    regularization_loss = t.sum(model.sigma, dim = 1) - model.latent_dim + mu_squared - t.log(t.prod(model.sigma, dim=1)) # shape: batch

    print(f"{reconstruction_loss.mean().item()=}")
    # print(f"{regularization_loss.mean().item()=}")


    return t.mean(reconstruction_loss + beta * regularization_loss)
#%%
train_one_epoch(beta_VAE_MNIST, training_dataloader, loss_bernoulli)

#%%
# print(f"{t.sum(beta_VAE_MNIST.sigma).shape=}")
# print(f"{beta_VAE_MNIST.latent_dim=}")
# print(f"{beta_VAE_MNIST.mu.shape=}")
# print(f"{ t.log(t.prod(beta_VAE_MNIST.sigma))=}")

#%%

#Loss function return: 64x1 then take the avg 






