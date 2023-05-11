#%%
import torch as t 
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader
from einops import rearrange, reduce, repeat
from arch import beta_VAE_chairs
from tqdm import tqdm
import math 
import random
import numpy as np

#%%
# Set seed to reproduce the same results (uncomment for experiments)
# I used this setup and got the same trainiing runs, however the pytorch documentation suggest some extra settings for the DataLoaders work_init_fn and generator

seed = 42
t.manual_seed(seed)
random.seed(seed)
t.use_deterministic_algorithms(True)
#%%
# Import MNIST data and split into batches
MNIST_data = datasets.MNIST(
    root='data',
    train = True,
    transform = Compose([ToTensor(), Resize((64, 64)),]),
    download= True)

training_dataloader = DataLoader(MNIST_data, batch_size=50, shuffle = True)

MNIST_data_test = datasets.MNIST(
    root='data',
    train = False,
    transform = Compose([ToTensor(), Resize((64, 64)),]),
    download= True)

test_dataloader = DataLoader(MNIST_data_test, shuffle = True)
#%%
#Create smaller dataset from MNIST
sample = random.sample(range(0,len(MNIST_data)), 10000)
MNIST_data_small = t.utils.data.Subset(MNIST_data, sample)

training_dataloader_small = DataLoader(MNIST_data_small, batch_size = 50, shuffle = True)
#%%

#Training the  net on small amount of data
#config = {'beta' : 1 }

beta = 1
def train_one_epoch(model, dataloader, loss_fun) -> float:
    t.autograd.set_detect_anomaly(True)
    optimizer = t.optim.Adam(model.parameters())

    total_loss = 0
    count = 0 
    for batch_x, batch_y in tqdm(dataloader):
        optimizer.zero_grad()
        decoder_output = model(batch_x)
        loss = loss_fun(model, batch_x, decoder_output, model.encoder_output, beta)
        total_loss += loss.item() * len(batch_x)
        loss.backward()
        optimizer.step()
        count += 1
        # if count > 50:
        #     break
    return (total_loss / len(dataloader.dataset))


#%%

def loss_bernoulli(model, input, decoder_output, encoder_output, beta) -> float:
    decoder_probs = t.sigmoid(decoder_output)
    decoder_output_scaled = t.where(t.abs(decoder_probs - 1.) < 10e-8, 1-10e-8, decoder_probs)
    decoder_output_scaled = t.where(t.abs(decoder_probs) < 10e-8, 10e-8, decoder_output_scaled)

    #Taylor expansion of log C close to 0.5
    log_C = t.where(decoder_output_scaled == 0.5, t.log(t.tensor((2.0))), decoder_output_scaled)
    log_C = t.where(t.abs(log_C - 0.5) < 10e-3, t.log(t.tensor(2)) + t.log(1+((1-2*decoder_output_scaled)**2)/3), log_C)
    mask = log_C == decoder_output_scaled
    decoder_output_processed = t.where(mask, decoder_output_scaled, 0.1)
    log_C = t.where(mask, t.log(2*t.atanh(1-2*decoder_output_processed)/(1-2*decoder_output_processed)), log_C) 
 
    
    #Reconstruction loss
    bce_loss = t.nn.BCEWithLogitsLoss(reduction = 'none')
    bce_loss_by_batch = reduce(bce_loss(input, decoder_output), 'b c h w -> b', 'sum')
    reconstruction_loss = (- t.sum(log_C, dim = (1,2,3)) + bce_loss_by_batch) #shape: (b,)

    #Regularization loss
    mu_squared = t.einsum('...i,...i -> ...', [model.mu, model.mu])
    regularization_loss = t.sum(model.sigma, dim = 1) - model.latent_dim + mu_squared - t.log(t.prod(model.sigma, dim=1)) #shape: (b,)

    return t.mean(reconstruction_loss + beta * regularization_loss)

#%%
#Training
beta_VAE_MNIST = beta_VAE_chairs(k = 10)
num_epochs = 20
train_losses = []
runner = tqdm(range(num_epochs))

for epoch in runner:
    train_losses.append(train_one_epoch(beta_VAE_MNIST, training_dataloader_small, loss_bernoulli))
plt.plot(train_losses, label='Train')
plt.legend()

#%%
#Examples
image1 = MNIST_data[0][0]
image2 = MNIST_data[1][0]

#images = t.stack((image1, image2), dim=0)

digit_images = [0]*10
counter = 0
for image, label in MNIST_data_test:
    if isinstance(digit_images[label], int):
        digit_images[label] = image
        counter += 1
    if counter == 10:
        break

images = t.stack(digit_images, dim = 0)

#%%
#Reconstructing examples
beta_VAE_MNIST.eval()
with t.no_grad():
    bernoulli_means = beta_VAE_MNIST(images)
beta_VAE_MNIST.train()
#%%
# Create a figure with 2 rows and 10 columns of subplots
fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4),  sharex=True, sharey=True,
                         gridspec_kw={"width_ratios": [1, 0.05] * 10})
axes = axes.flatten()

# Loop over each subplot and plot the data
for i, ax in enumerate(axes): #enumerate(axes):
    if i < 10:
        original_img = images[i].detach().reshape(64, 64, 1)
        im = ax.imshow(original_img, vmin = 0, vmax = 1)
    else:
        reconstructed_img = beta_VAE_MNIST.reconstruct()[i-10].detach().reshape(64,64,1)
        im = ax.imshow(reconstructed_img, vmin = 0, vmax = 1)

    #ax.plot(x, y * (i+1))
    #ax.set_title(f"Plot {i+1}")

# Adjust the layout of the subplots and show the figure
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(im, ax=axes.ravel().tolist())
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

#cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
#plt.colorbar(im, cax=cax, **kw)
plt.subplots_adjust(wspace=0, hspace=0)
#plt.tight_layout()
plt.show()


#%%
index = 1
original_img = images[index].detach().reshape(64, 64, 1)
reconstructed_img = beta_VAE_MNIST.reconstruct()[index].detach().reshape(64,64,1)
#%%
plt.imshow(original_img, vmin = 0, vmax = 1)
plt.colorbar()
#%%
plt.imshow(reconstructed_img, vmin = 0, vmax = 1)
plt.colorbar()
#%%
#Save model (Remember to add a new name for each new model)
model_name = 'basic_01.pt'
path = '/workspaces/beta-vae-replication/saved_models/' + model_name
t.save(beta_VAE_MNIST.state_dict(), path)
