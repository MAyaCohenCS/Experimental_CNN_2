from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

dataroot = "data/celeba/"

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Create the dataloader

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def half_face_mask(image):
    channels, height, width =  im_shape = image.shape
    left_half_bound = height//2
    image[:, :, :left_half_bound] = 0
    return image 

def random_mask(image, err_rate):
    im_shape = image.shape
    image = image.reshape(-1)
    pixels_num = len(image)
    noise_indxs = np.random.choice(pixels_num, int(pixels_num * err_rate), replace=False)
    image[noise_indxs] = 0
    image = image.reshape(im_shape)

    return image

def hide_eyes_mask(image):
    channels, height, width = image.shape
    inner_3rd_bounds = [width//3, 2*width//3]
    inner_5th_bounds = [2*height//5, 3*height//5]
    image[:, inner_5th_bounds[0]:inner_5th_bounds[1], inner_3rd_bounds[0]:inner_3rd_bounds[1]] = 0
    return image 

# Plot some training images
real_batch = next(iter(dataloader))
for i in range(len(real_batch[0])):
    random_mask(real_batch[0][i], 0.3)
plt.figure(figsize=(4,4))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:16], padding=2, normalize=True).cpu(),(1,2,0)))

plt.show()

