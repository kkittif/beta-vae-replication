#%%
import torch as t 
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import MNIST data and split into batches
MNIST_data = datasets.MNIST(
    root='data',
    train = True,
    transform = Compose([ToTensor(), Resize((64, 64)),]),
    download= True)

training_dataloader = DataLoader(MNIST_data, batch_size=64, shuffle = True)

#%% 

# Visualise the data 
plt.imshow(MNIST_data[0][0].reshape((64, 64, 1)))

from arch import beta_VAE_chairs

#%%
beta_VAE_MNIST = beta_VAE_chairs(k = 10)

# for params in beta_VAE_MNIST.decoder[6].parameters():
#     print(f"{params.shape=}")
#     print(f"{params.max()=}")
#     print(f"{params.min()=}")
#     print(f"{params.mean()=}")
#     print(f"{params.std()=}")

#%%

#Testing a forward pass
image1 = MNIST_data[0][0]
image2 = MNIST_data[1][0]

images = t.stack((image1, image2), dim=0)
bernoulli_means = beta_VAE_MNIST(images)

# print(bernoulli_means)
print(bernoulli_means)
print(t.max(bernoulli_means))
print(t.min(bernoulli_means))

plt.imshow(bernoulli_means[0].detach().reshape(64, 64, 1))


# %%
plt.imshow(bernoulli_means[1].detach().reshape(64, 64, 1))

#%%

#Training the  net on small amount of data
#config = {'beta' : 1 }

beta = 1
def train_one_epoch(model, dataloader, loss_fun) -> float:
    optimizer = t.optim.Adam(model.parameters())
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        decoder_output = model(batch_x)
        loss = loss_fun(model, batch_x, decoder_output, model.encoder_output, beta)
        total_loss += loss * len(batch_x)
        loss.backward()
        optimizer.step()
        break
    return (total_loss / len(dataloader.dataset))   #.item()

def loss_bernoulli(model, input, decoder_output, encoder_output, beta) -> float:
    C = t.where(decoder_output != 0.5, 2*t.atanh(1-2*decoder_output)/(1-2*decoder_output), 2) # TODO: What is the decoder output is really close to 0.5? Taylor-expansion
    reconstruction_loss = -t.sum(C + input*t.log(decoder_output) + (1 - input) * t.log(1 - decoder_output)).item()
    
    regularization_loss = (t.sum(model.sigma) - model.latent_dim + model.mu @ model.mu.T - t.log(t.prod(model.sigma))).item()

    return reconstruction_loss + beta * regularization_loss
#%%
train_one_epoch(beta_VAE_MNIST, training_dataloader, loss_bernoulli)

#%%
print(f"{t.sum(beta_VAE_MNIST.sigma).shape=}")
print(f"{beta_VAE_MNIST.latent_dim=}")
print(f"{beta_VAE_MNIST.mu.shape=}")
print(f"{ t.log(t.prod(beta_VAE_MNIST.sigma))=}")

#%%

#Loss function return: 64x1 then take the avg 






