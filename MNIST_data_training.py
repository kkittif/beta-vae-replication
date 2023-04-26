#%%
import torch as t 
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


MNIST_data = datasets.MNIST(
    root='data',
    train = True,
    transform = Compose([ToTensor(), Resize((64, 64)),]),
    download= True)

# print(MNIST_data[0][0])

#%% 
plt.imshow(MNIST_data[0][0].reshape((64, 64, 1)))

from arch import beta_VAE_chairs

#%%
beta_VAE_MNIST = beta_VAE_chairs(k = 10)

#%%
bernoulli_means = beta_VAE_MNIST(MNIST_data[0][0])

# print(bernoulli_means)
print(bernoulli_means)
print(t.max(bernoulli_means))
print(t.min(bernoulli_means))

plt.imshow(bernoulli_means.detach().reshape(64, 64, 1))

# %%
