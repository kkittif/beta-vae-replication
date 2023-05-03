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

#%% 
from arch import beta_VAE_chairs
beta_VAE_MNIST = beta_VAE_chairs(k = 10)

#%%
#Testing a forward pass
image1 = MNIST_data[0][0]
image2 = MNIST_data[1][0]

images = t.stack((image1, image2), dim=0)
bernoulli_means = beta_VAE_MNIST(images)

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
        if t.isnan(loss):
            print('Loss was nan')
            break
        total_loss += loss.item() * len(batch_x)
        loss.backward()
        optimizer.step()
        count += 1
        if count > 200:
            break
        
    return (total_loss / len(dataloader.dataset))   #.item()


#%%

def loss_bernoulli(model, input, decoder_output, encoder_output, beta) -> float:
    decoder_probs = t.sigmoid(decoder_output)
    decoder_output_scaled = t.where(t.abs(decoder_probs - 1.) < 10e-8, 1-10e-8, decoder_probs)
    decoder_output_scaled = t.where(t.abs(decoder_probs) < 10e-8, 10e-8, decoder_output_scaled)

    #Taylor expansion of log C close to 0.5
    log_C = t.where(decoder_output_scaled == 0.5, t.log(t.tensor((2.0))), decoder_output_scaled)
    log_C = t.where(t.abs(log_C - 0.5) < 10e-3, t.log(t.tensor(2)) + t.log(1+((1-2*decoder_output_scaled)**2)/3), log_C)
    mask = log_C == decoder_output_scaled
    log_C = t.where(mask, t.log(2*t.atanh(1-2*decoder_output_scaled)/(1-2*decoder_output_scaled)), log_C) 
 
    
    #Reconstruction loss
    bce_loss = t.nn.BCEWithLogitsLoss(reduction = 'none')
    bce_loss_by_batch = reduce(bce_loss(input, decoder_output), 'b c h w -> b', 'sum')
    reconstruction_loss = - (t.sum(log_C, dim = (1,2,3)) + bce_loss_by_batch) #shape: (b,)

    #Regularization loss
    mu_squared = t.einsum('...i,...i -> ...', [model.mu, model.mu])
    regularization_loss = t.sum(model.sigma, dim = 1) - model.latent_dim + mu_squared - t.log(t.prod(model.sigma, dim=1)) #shape: (b,)

    return t.mean(reconstruction_loss + beta * regularization_loss)

#%%
#Visualize training loss
num_epoch = 1
train_losses = []
for epoch in range(num_epoch):
    train_losses.append(train_one_epoch(beta_VAE_MNIST, training_dataloader, loss_bernoulli))
#plt.plot(train_losses, label='Train')
#plt.legend()

#%%
#Print one image's reconstruction
plt.imshow(images[1].detach().reshape(64, 64, 1))

#%%
with t.no_grad():
    bernoulli_means = beta_VAE_MNIST(images)
    plt.imshow(beta_VAE_MNIST.reconstruct(sample = True)[0].detach().reshape(64,64,1))

    #plt.imshow(bernoulli_means[0].detach().reshape(64, 64, 1))
    print(t.min(beta_VAE_MNIST.reconstruct(sample = False)[1]))

#%%
target = t.ones([10, 64], dtype=t.float32)  # 64 classes, batch size = 10
output = t.full([10, 64], 1.5)  # A prediction (logit)
pos_weight = t.ones([64])  # All weights are equal to 1
criterion = t.nn.BCEWithLogitsLoss(reduction = 'none')
criterion(output, target).shape  # -log(sigmoid(1.5))
