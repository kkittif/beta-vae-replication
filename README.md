# beta-vae-replication

In this project we implemented a selected beta-VAE architecture from the original paper: https://openreview.net/pdf?id=Sy2fzU9gl

Below is a summary, our more detailed project write up on implementation details and lessons learned is found [here](https://brindle-wallet-5cd.notion.site/VAE-implementation-intuition-results-and-lessons-learned-1a3d172365db4fe0b0e274f940dcd29a)

# VAE implementation details

## Architecture overview:
A VAE is comprised of two networks: an encoder and a decoder. The encoder compresses an input $x$ into a smaller vector, $z$. The decoder creates a reconstruction of $x$ from $z$ 
![Untitled (18)](https://github.com/kkittif/beta-vae-replication/assets/46658522/13d2ba39-b149-4225-bf60-e532ecf5a405)

The more detailed picture is as follows (see our linked write up for explanatory details):

![Untitled (13)](https://github.com/kkittif/beta-vae-replication/assets/46658522/a8eef280-f521-4094-87e8-2689348a5d64)

## The encoder:
Zooming in, the encoder architecture is shown. It is primarily comprised of convolutional layers which copmress the input image.
![Untitled (16)](https://github.com/kkittif/beta-vae-replication/assets/46658522/d9025745-d177-4605-9360-13d9b7a3b075)

## The decoder:
The decoder is roughly the reverse of the encoder, with convolutions replaced by transposed convolutions.
![Untitled (17)](https://github.com/kkittif/beta-vae-replication/assets/46658522/3ae666fa-b037-4c71-82e9-f2ef71319776)

## Training and results
We trained our model for 100 epochs with $\beta = 1$ on 10000 images from the MNIST dataset using a batchsize of 50. The follwoing displays the reconstruction of some digits using after selected epochs of training the model: 

![image](https://github.com/kkittif/beta-vae-replication/assets/46658522/64c5e7ee-ab6d-438d-8d23-13189b0b420a)

