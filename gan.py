import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.datasets import MNIST

from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader


from tqdm import tqdm
import numpy as np

import imageio
import matplotlib
from matplotlib import pyplot as plt

matplotlib.style.use("ggplot")


# to create real labels (1s)
def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)


# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)


# function to create the noise vector
def create_noise(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)


def save_generator_image(image, path):
    save_image(image, path)


# learning parameters
batch_size = 16
epochs = 1
sample_size = 64  # fixed sample size
nz = 128  # latent vector size
k = 1  # number of steps to apply to the discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
to_pil_image = transforms.ToPILImage()


# Download training dataset
# dataset = MNIST(root='data/', download=True)

train_data = datasets.MNIST(
    root="data/", train=True, download=True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.n_input_h = 5
        self.n_input_w = 5
        self.n_input_channels = 32

        self.fc1 = nn.Linear(nz, 128)
        self.fc2 = nn.Linear(128, 800)

        self.tconv1 = nn.ConvTranspose2d(32, 16, 3, 2)
        self.tconv2 = nn.ConvTranspose2d(16, 1, 3, 2)


    def forward(self, x):

        x = x.view(-1, self.nz)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = x.view(-1, self.n_input_channels, self.n_input_h, self.n_input_w)
        x = F.relu(self.tconv1(F.pad(x, pad=(0,1,0,1))))
        x = F.pad(F.relu(self.tconv2(x)),pad=(0,1,0,1))
        x = torch.tanh(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input_h = 28
        self.n_input_w = 28
        self.n_input_channels = 1

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        x = x.view(-1, self.n_input_channels, self.n_input_h, self.n_input_w)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)
print("##### GENERATOR #####")
print(generator)
print("######################")
print("\n##### DISCRIMINATOR #####")
print(discriminator)
print("######################")

optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# loss function
criterion = nn.BCELoss()

losses_g = []  # to store generator loss after each epoch
losses_d = []  # to store discriminator loss after each epoch
images = []  # to store images generatd by the generator


# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)
    optimizer.zero_grad()
    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    return loss_real + loss_fake


# function to train the generator network
def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)
    optimizer.zero_grad()
    output = discriminator(data_fake)
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    return loss


# create the noise vector
noise = create_noise(sample_size, nz)

generator.train()
discriminator.train()

for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for bi, data in tqdm(
        enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)
    ):
        image, _ = data
        image = image.to(device)
        b_size = len(image)
        # run the discriminator for k number of steps
        for step in range(k):
            data_fake = generator(create_noise(b_size, nz)).detach()
            data_real = image
            # train the discriminator network
            loss_d += train_discriminator(optim_d, data_real, data_fake)
        data_fake = generator(create_noise(b_size, nz))
        # train the generator network
        loss_g += train_generator(optim_g, data_fake)
    # create the final fake image for the epoch
    generated_img = generator(noise).cpu().detach()
    # make the images as grid
    generated_img = make_grid(generated_img)
    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"../outputs/gen_img{epoch}.png")
    images.append(generated_img)
    epoch_loss_g = loss_g / bi  # total generator loss for the epoch
    epoch_loss_d = loss_d / bi  # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)

    print(f"Epoch {epoch} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

print("DONE TRAINING")
torch.save(generator.state_dict(), "../outputs/generator.pth")

# save the generated images as GIF file
imgs = [np.array(to_pil_image(img)) for img in images]
imageio.mimsave("../outputs/generator_images.gif", imgs)


# save the generated images as GIF file
imgs = [np.array(to_pil_image(img)) for img in images]
imageio.mimsave("../outputs/generator_images.gif", imgs)
