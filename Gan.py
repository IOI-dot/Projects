import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils

DATA_DIR = r"C:\Users\Omar\Downloads\archive (20)"
BATCH_SIZE = 64
IMG_SIZE = 64
EPOCHS = 50
LR = 0.0002
LATENT_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print(f"{len(dataset)} images, classes: {dataset.classes}")

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            # input is Z: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

netG = Generator(LATENT_DIM).to(DEVICE)
netD = Discriminator().to(DEVICE)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)
if __name__ == "__main__":
    for epoch in range(EPOCHS):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)

            # Labels
            real_labels = torch.full((batch_size, 1), 1., device=DEVICE)
            fake_labels = torch.full((batch_size, 1), 0., device=DEVICE)

            #Train Discriminator
            netD.zero_grad()
            output_real = netD(real_images)
            lossD_real = criterion(output_real, real_labels)

            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())
            lossD_fake = criterion(output_fake, fake_labels)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            #Train Generator
            netG.zero_grad()
            output = netD(fake_images)
            lossG = criterion(output, real_labels)  # trick D
            lossG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{i}/{len(dataloader)}] "
                      f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"fake_epoch_{epoch+1}.png", normalize=True)

    torch.save(netG.state_dict(), "dcgan_generator.pth")
    torch.save(netD.state_dict(), "dcgan_discriminator.pth")
    print("Models saved: dcgan_generator.pth, dcgan_discriminator.pth")
