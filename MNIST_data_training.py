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
# Set any constant here
config = {'beta' : 1, 
          'epochs' : 20,
          'batch_size' : 50,
          'training_size' : 5000,
          'latent_space' : 10,
          'random_seed' : 42}


#%%
# Set seed to reproduce the same results (uncomment for experiments)
# I used this setup and got the same trainiing runs, however the pytorch documentation suggest some extra settings for the DataLoaders work_init_fn and generator

seed = config['random_seed']
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

training_dataloader = DataLoader(MNIST_data, batch_size = config['batch_size'], shuffle = True)

MNIST_data_test = datasets.MNIST(
    root='data',
    train = False,
    transform = Compose([ToTensor(), Resize((64, 64)),]),
    download= True)

test_dataloader = DataLoader(MNIST_data_test, shuffle = True)
#%%
#Create smaller dataset from MNIST
sample = random.sample(range(0,len(MNIST_data)), config['training_size'])
MNIST_data_small = t.utils.data.Subset(MNIST_data, sample)

training_dataloader_small = DataLoader(MNIST_data_small, batch_size = config['batch_size'], shuffle = True)

#%%
#Examples
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
# Function plotting the reconstructions of digit examples

def plot_recon_digits(images):

    beta_VAE_MNIST.eval()
    with t.no_grad():
        bernoulli_means = beta_VAE_MNIST(images)
    beta_VAE_MNIST.train()

    # Create a figure with 2 rows and 10 columns of subplots
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    axes = axes.flatten()

    # Loop over each subplot and plot the data
    for i, ax in enumerate(axes): #enumerate(axes):
        if i < 10:
            original_img = images[i].detach().reshape(64, 64, 1)
            im = ax.imshow(original_img, vmin = 0, vmax = 1)
        else:
            reconstructed_img = beta_VAE_MNIST.reconstruct()[i-10].detach().reshape(64,64,1)
            im = ax.imshow(reconstructed_img, vmin = 0, vmax = 1)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    plt.show()


#%%
def train_one_epoch(model, dataloader, loss_fun) -> float:
    t.autograd.set_detect_anomaly(True)
    optimizer = t.optim.Adam(model.parameters())

    total_loss = 0
    total_rec_loss = 0
    total_reg_loss = 0
    count = 0 
    for batch_x, batch_y in tqdm(dataloader):
        optimizer.zero_grad()
        decoder_output = model(batch_x)
        losses = loss_fun(model, batch_x, decoder_output, model.encoder_output, config['beta'])
        loss = losses[0]
        total_loss += loss.item() * len(batch_x)
        total_rec_loss += losses[1].item() * len(batch_x)
        total_reg_loss += losses[2].item() * len(batch_x)
        loss.backward()
        optimizer.step()
        count += 1
    return ((total_loss / len(dataloader.dataset)), total_rec_loss / len(dataloader.dataset), total_reg_loss / len(dataloader.dataset) )

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
    bce_loss_by_batch = reduce(bce_loss(decoder_output, input), 'b c h w -> b', 'sum')
    reconstruction_loss = (- t.sum(log_C, dim = (1,2,3)) + bce_loss_by_batch) #shape: (b,)

    #Regularization loss
    mu_squared = t.einsum('...i,...i -> ...', [model.mu, model.mu])
    regularization_loss = 0.5*(t.sum(model.sigma, dim = 1) - model.latent_dim + mu_squared - t.log(t.prod(model.sigma, dim=1))) #shape: (b,)

    return (t.mean(reconstruction_loss + beta * regularization_loss), t.mean(reconstruction_loss), t.mean(beta*regularization_loss))

#%%
#Training
beta_VAE_MNIST = beta_VAE_chairs(k = config['latent_space'])
num_epochs = config['epochs']
train_losses = []
rec_losses = []
reg_losses = []
runner = tqdm(range(num_epochs))
#%%
for epoch in runner:
    three_losses = train_one_epoch(beta_VAE_MNIST, training_dataloader_small, loss_bernoulli)
    train_losses.append(three_losses[0])
    rec_losses.append(three_losses[1])
    reg_losses.append(three_losses[2])
    print('Epoch: ', epoch)
    plot_recon_digits(images)

#%%
plt.plot(train_losses, label='Training')
plt.plot(reg_losses, label='Regularization')
plt.plot(rec_losses, label='Reconstruction')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
#%%
print(reg_losses)
plt.plot(reg_losses)
plt.title('Regularization loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#%%
#Reconstructing examples
beta_VAE_MNIST.eval()
with t.no_grad():
    bernoulli_means = beta_VAE_MNIST(images)
beta_VAE_MNIST.train()

#%%
#Save model (Remember to add a new name for each new model)
save_model = False
if save_model:
    model_name = 'basic_02.pt'
    path = '/workspaces/beta-vae-replication/saved_models/' + model_name
    t.save(beta_VAE_MNIST.state_dict(), path)

#%%
#If only training on two images, we can use these to see what the training example is reconstructed as
index = 1
img_1 = MNIST_data_small[1][0]
img_0 = MNIST_data_small[0][0]
two_digits = t.stack((img_1, img_0))
beta_VAE_MNIST.eval()
with t.no_grad():
    bernoulli_means = beta_VAE_MNIST(two_digits)
beta_VAE_MNIST.train()
imf = beta_VAE_MNIST.reconstruct()[index].detach().reshape(64,64,1)
plt.imshow(imf, vmin = 0, vmax = 1)

#%%
# Playground

#Plotting the weights of the last unconvolutional layer
last_unconv_params = dict(beta_VAE_MNIST.named_parameters())['decoder.14.weight'].detach()
x = rearrange(last_unconv_params, 'b c w h -> (b w) h c')
print(plt.imshow(x))