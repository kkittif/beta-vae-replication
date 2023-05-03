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
        if t.isnan(loss):
            print('Loss was nan')
            break
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
        if count > 20:
            break
        
    return (total_loss / len(dataloader.dataset))   #.item()

def loss_bernoulli(model, input, decoder_output, encoder_output, beta) -> float:
    print(f"{t.min(decoder_output).item()=}")
    decoder_output_scaled = t.where(t.abs(decoder_output - 1.) < 10e-8, 1-10e-8, decoder_output)
    decoder_output_scaled = t.where(t.abs(decoder_output) < 10e-8, 10e-8, decoder_output_scaled)

    C = t.where(t.abs(decoder_output_scaled - 0.5) > 10e-4, 2*t.atanh(1-2*decoder_output_scaled)/(1-2*decoder_output_scaled), 2.0)
    # if t.min(decoder_output_scaled).item() < 10e-5:
    #     print(f"{t.min(decoder_output_scaled).item()=}")

    
    # if t.max(decoder_output_scaled).item() > 1-10e-5:
    #     print(f"{t.max(decoder_output_scaled).item()=}")
    #C = t.where((decoder_output_scaled != 0.5), 2*t.atanh(1-2*decoder_output_scaled)/(1-2*decoder_output_scaled), 2.0) 
    #print(f"{(C)=}")
    print(f"{t.max(C)=}")
    # if t.any(1-2*0.99*decoder_output_scaled == 0.):
    #     print("ZERO")

    print(t.any((1-2*decoder_output_scaled) == 0 ))
    print(t.any((1-2*decoder_output_scaled) == 1 ))
    print(t.max(t.atanh(1-2*decoder_output_scaled)))
    print(t.min(t.atanh(1-2*decoder_output_scaled)))

    # TODO: What is the decoder output is really close to 0.5? Taylor-expansion
    reconstruction_loss = -reduce((t.log(C) + input*t.log(decoder_output_scaled) + (1 - input) * t.log(1 - decoder_output_scaled)), 'b c h w -> b ', reduction = 'sum') # shape: batch
    # if t.any(t.isnan(reconstruction_loss)):
    #     print("New loss calc")
    #     print(t.min(t.atanh(1-2*0.99*decoder_output_scaled)))
    print(f"{t.min(decoder_output_scaled).item()=}")
    print(f"{t.max(decoder_output_scaled).item()=}")

    mu_squared = t.einsum('...i,...i -> ...', [model.mu, model.mu])
    #print(regularization_loss.dtype)
    regularization_loss = t.sum(model.sigma, dim = 1) - model.latent_dim + mu_squared - t.log(t.prod(model.sigma, dim=1)) # shape: batch

    # if regularization_loss.dtype != t.float32 or reconstruction_loss.dtype != t.float32:
    print(f"{regularization_loss.mean().item()=}")
    print(f"{reconstruction_loss.mean().item()=}")
        #print(t.any(regularization_loss == reconstruction_loss))
    #if t.any(t.isnan(regularization_loss)):
     #   print(f"{regularization_loss.mean().item()=}")
    #print(f"{t.min(t.prod(model.sigma, dim=1))=}")

    return t.mean(reconstruction_loss + beta * regularization_loss)
#%%

def loss_bernoulli_BCE(model, input, decoder_output, encoder_output, beta) -> float:
    #Original C calculation
    decoder_probs = t.sigmoid(decoder_output)
    #decoder_output_scaled = t.where(t.abs(decoder_probs - 1.) < 10e-8, 1-10e-8, decoder_probs)
    #decoder_output_scaled = t.where(t.abs(decoder_probs) < 10e-8, 10e-8, decoder_output_scaled)

    #C = t.where(t.abs(decoder_output_scaled - 0.5) > 10e-4, 2*t.atanh(1-2*decoder_output_scaled)/(1-2*decoder_output_scaled), 2.0)

    #C with Taylor expansion
    log_C = t.log(t.tensor(2)) + t.log(1+((1-2*decoder_probs)**2)/3)

    #Reconstruction loss
    #decoder_probs = t.sigmoid(decoder_output)
    #C = t.where(t.abs(decoder_probs - 0.5) > 10e-4, 2*t.atanh(1-2*decoder_probs)/(1-2*decoder_probs), 2.0)
    print(f"{t.min(decoder_output).item()=}")
    print(f"{t.max(decoder_output).item()=}")
    bce_loss = t.nn.BCEWithLogitsLoss(reduction = 'none')
    bce_loss_by_batch = reduce(bce_loss(input, decoder_output), 'b c h w -> b', 'sum')
    print(f"{bce_loss_by_batch.min()=}")
    print(f"{bce_loss_by_batch.max()=}")
    #print(f"{t.sum(t.log(C))=}")
    print(f"{t.sum(log_C)=}")

    #reconstruction_loss = - (t.sum(t.log(C), dim = (1,2,3)) + bce_loss_by_batch) #shape: (b,)
    reconstruction_loss = - (t.sum(log_C, dim = (1,2,3)) + bce_loss_by_batch) #shape: (b,)

    #Regularization loss
    mu_squared = t.einsum('...i,...i -> ...', [model.mu, model.mu])
    regularization_loss = t.sum(model.sigma, dim = 1) - model.latent_dim + mu_squared - t.log(t.prod(model.sigma, dim=1)) # shape: batch
    print(f"{reconstruction_loss.mean().item()=}")
    print(f"{regularization_loss.mean().item()=}")

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
#Print one images reconstruction
plt.imshow(images[0].detach().reshape(64, 64, 1))
#%%
with t.no_grad():
    bernoulli_means = beta_VAE_MNIST(images)
    plt.imshow(beta_VAE_MNIST.reconstruct(sample = True)[0].detach().reshape(64,64,1))
    #plt.imshow(bernoulli_means[0].detach().reshape(64, 64, 1))



#%%
# print(f"{t.sum(beta_VAE_MNIST.sigma).shape=}")
# print(f"{beta_VAE_MNIST.latent_dim=}")
# print(f"{beta_VAE_MNIST.mu.shape=}")
# print(f"{ t.log(t.prod(beta_VAE_MNIST.sigma))=}")

#%%

target = t.ones([10, 64], dtype=t.float32)  # 64 classes, batch size = 10
output = t.full([10, 64], 1.5)  # A prediction (logit)
pos_weight = t.ones([64])  # All weights are equal to 1
criterion = t.nn.BCEWithLogitsLoss(reduction = 'none')
criterion(output, target).shape  # -log(sigmoid(1.5))

#%%
x = t.tensor(-1)
t.atanh(x)