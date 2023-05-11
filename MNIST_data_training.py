#%%
import torch as t 
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops import rearrange, reduce, repeat
import math 
import random

#%%
#Set seed to reproduce the same results (uncomment for experiments)
# I used this setup and got the same trainiing runs, however the pytorch documentation suggest some extra settings for the DataLoaders work_init_fn and generator
# seed = 42
# t.manual_seed(seed)
# random.seed(seed)
# t.use_deterministic_algorithms(True)
#%%
# Import MNIST data and split into batches
MNIST_data = datasets.MNIST(
    root='data',
    train = True,
    transform = Compose([ToTensor(), Resize((64, 64)),]),
    download= True)

training_dataloader = DataLoader(MNIST_data, batch_size = 50, shuffle = True)

#%%
#Create smaller dataset from MNIST
sample = random.sample(range(0,len(MNIST_data)), 10000)
MNIST_data_small = t.utils.data.Subset(MNIST_data, sample)

training_dataloader_small = DataLoader(MNIST_data_small, batch_size = 50, shuffle = True)
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
    t.autograd.set_detect_anomaly(True)
    optimizer = t.optim.Adam(model.parameters())
    total_loss = 0
    count = 0 
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        decoder_output = model(batch_x)
        print(t.any(decoder_output == 0.))
        loss = loss_fun(model, batch_x, decoder_output, model.encoder_output, beta)
        print(f"{loss=}")
        if t.isnan(loss):
            print('Loss was nan')
            break
        total_loss += loss.item() * len(batch_x)
        loss.backward()
        optimizer.step()
        count += 1
        if count > 50:
            break
    return (total_loss / len(dataloader.dataset))   #.item()


#%%

def loss_bernoulli(model, input, decoder_output, encoder_output, beta) -> float:
    decoder_probs = t.sigmoid(decoder_output)
    decoder_output_scaled = t.where(t.abs(decoder_probs - 1.) < 10e-8, 1-10e-8, decoder_probs)
    decoder_output_scaled = t.where(t.abs(decoder_probs) < 10e-8, 10e-8, decoder_output_scaled)

    #Taylor expansion of log C close to 0.5
    log_C = t.where(decoder_output_scaled == 0.5, t.log(t.tensor((2.0))), decoder_output_scaled)
    print(log_C.grad_fn, 1)
    mask_1 = decoder_output_scaled != 0.5
    log_C = t.where(t.abs(log_C - 0.5) < 10e-3, t.log(t.tensor(2)) + t.log(1+((1-2*decoder_output_scaled)**2)/3), log_C)
    print(log_C.grad_fn, 2)
    mask_2 =  t.abs(decoder_output_scaled - 0.5) >= 10e-3
    #mask = log_C == decoder_output_scaled
    mask = t.logical_and(mask_1, mask_2) 
    
    print(f"{t.any(2*t.atanh(1-2*decoder_output_scaled) == 0. )=}")
    print(f"{t.any((1-2*log_C) == 0.5 )=}")
    print(f"{t.any(t.logical_and((1-2*decoder_output_scaled) == 0.5, mask ))=}") #If this is True then our mask does not work. This has to be false for the mask to work

    decoder_output_processed = t.where(mask, decoder_output_scaled, 0.1)


    #exact_value = )
    #print(exact_value.grad_fn)
    log_C = t.where(mask, t.log(2*t.atanh(1-2*decoder_output_processed)/(1-2*decoder_output_processed)), log_C)
    print(log_C.grad_fn, 3)
 
    
    #Reconstruction loss
    bce_loss = t.nn.BCEWithLogitsLoss(reduction = 'none')
    bce_loss_by_batch = reduce(bce_loss(input, decoder_output), 'b c h w -> b', 'sum')
    reconstruction_loss = (- t.sum(log_C, dim = (1,2,3)) + bce_loss_by_batch) #shape: (b,)
    print(reconstruction_loss.grad_fn)


    #Regularization loss
    mu_squared = t.einsum('...i,...i -> ...', [model.mu, model.mu])

    regularization_loss = t.sum(model.sigma, dim = 1) - model.latent_dim + mu_squared - t.log(t.prod(model.sigma, dim=1)) #shape: (b,)
    print(regularization_loss.grad_fn)
    return t.mean(reconstruction_loss + beta * regularization_loss)

#%%
#Trivial loss to debug our backprop
# model = beta_VAE_chairs(k = 10)
# optimizer = t.optim.Adam(model.parameters())
# optimizer.zero_grad()


# total_loss = 0
# batch_x, batch_y = next(iter(training_dataloader_small)) 
# batch_x = batch_x.requires_grad_()
# decoder_output = model(batch_x)

# loss = t.sum(decoder_output[2,:,:,:], dim = (0,1,2))
# grads0 = t.autograd.grad(outputs = loss, inputs= batch_x)
# print(grads0)
# #loss.backward(retain_graph = True)

# #grads = t.autograd.grad(outputs = loss, inputs= batch_x)
# #%%

# print(grads[0][0])
# #print(batch_x[0].grad)
# #%%
# print(grads[0][:,:,10,:])

#%%
#Train
num_epoch = 1
train_losses = []
for epoch in range(num_epoch):
    train_losses.append(train_one_epoch(beta_VAE_MNIST, training_dataloader_small, loss_bernoulli))
#%%
#Visualize training loss
plt.plot(train_losses, label='Train')
plt.legend()

#%%
#Print one image's reconstruction
plt.imshow(images[1].detach().reshape(64, 64, 1))

#%%
with t.no_grad():
    bernoulli_means = beta_VAE_MNIST(images)
    plt.imshow(beta_VAE_MNIST.reconstruct()[0].detach().reshape(64,64,1))

    #plt.imshow(bernoulli_means[0].detach().reshape(64, 64, 1))
    print(t.min(beta_VAE_MNIST.reconstruct()[0]))
#%%
index = 1
original_img = images[index].detach().reshape(64, 64, 1)
reconstructed_img = beta_VAE_MNIST.reconstruct()[index].detach().reshape(64,64,1)
difference = t.abs(original_img - reconstructed_img)
#%%
plt.imshow(original_img, vmin = 0, vmax = 1)
plt.colorbar()
#%%
plt.imshow(reconstructed_img, vmin = 0, vmax = 1)
plt.colorbar()
#%%
target = t.ones([10, 64], dtype=t.float32)  # 64 classes, batch size = 10
output = t.full([10, 64], 1.5)  # A prediction (logit)
pos_weight = t.ones([64])  # All weights are equal to 1
criterion = t.nn.BCEWithLogitsLoss(reduction = 'none')
criterion(output, target).shape  # -log(sigmoid(1.5))
#%%
#Experimenting with backward hooks
backward_values = {}
def hook_fn_backward(module, inp_grad, out_grad):
    backward_values[module] = {}
    backward_values[module]["input"] = inp_grad
    backward_values[module]["output"] = out_grad

model = beta_VAE_chairs()

modules = model.named_children()
print(modules)
#%%
for name, module in modules:
    module.register_backward_hook(hook_fn_backward)

#%%
num_epoch = 1
train_losses = []
for epoch in range(num_epoch):
    train_losses.append(train_one_epoch(model, training_dataloader_small, loss_bernoulli))

#%%
#print(backward_values)
print(modules)
for name, module in modules:
    print(name)
    print(backward_values[module]['input'])


#%%
#Testing t.where
tensor = t.randn(10,10)
tensor[0,5] = 0
mask = tensor != 0
div = t.where(mask, 1/tensor, )
print(div)