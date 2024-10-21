import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

class GAN3DModelGenerator:
    def __init__(self, z_dim=100, lr=0.0002, batch_size=64, num_epochs=100, use_synthetic_data=True):
        # Define Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.z_dim = z_dim
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_synthetic_data = use_synthetic_data

        # Create models
        self.generator = self.Generator(input_dim=z_dim).to(self.device)
        self.discriminator = self.Discriminator().to(self.device)

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # Loss function
        self.criterion = nn.BCELoss()

    class Generator(nn.Module):
        def __init__(self, input_dim=100, output_dim=3):
            super(GAN3DModelGenerator.Generator, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)  # Assuming 3D coordinates output
            )

        def forward(self, z):
            return self.fc(z)

    class Discriminator(nn.Module):
        def __init__(self, input_dim=3):
            super(GAN3DModelGenerator.Discriminator, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.fc(x)

    def train(self):
        # Training loop for GAN
        for epoch in range(self.num_epochs):
            # Generate random latent vectors
            z = torch.randn(self.batch_size, self.z_dim).to(self.device)
            generated_data = self.generator(z)

            # Train Discriminator
            real_labels = torch.ones(self.batch_size, 1).to(self.device)
            fake_labels = torch.zeros(self.batch_size, 1).to(self.device)

            self.discriminator.zero_grad()
            if self.use_synthetic_data:
                # Use synthetic real data if ShapeNet is not available
                fake_data = torch.randn(self.batch_size, 3).to(self.device)  # Simulated real 3D data
            else:
                # Load real data from ShapeNet (not implemented here)
                fake_data = torch.randn(self.batch_size, 3).to(self.device)  # Placeholder

            real_loss = self.criterion(self.discriminator(generated_data.detach()), fake_labels)  # Generated data is fake
            fake_loss = self.criterion(self.discriminator(fake_data), real_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer_D.step()

            # Train Generator
            self.generator.zero_grad()
            g_loss = self.criterion(self.discriminator(generated_data), real_labels)
            g_loss.backward()
            self.optimizer_G.step()

            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.num_epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}')

    def generate_and_visualize(self):
        # Generate and visualize a basic 3D shape after training
        with torch.no_grad():
            z = torch.randn(1, self.z_dim).to(self.device)
            generated_model = self.generator(z).cpu().numpy().flatten()
            print(f"Generated 3D model: {generated_model}")

            # Simple scatter plot for the generated shape using Streamlit
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(generated_model[0], generated_model[1], generated_model[2], c='r', marker='o')
            st.pyplot(fig)

# Post-design generation prompt
if __name__ == "__main__":
    gan_model = GAN3DModelGenerator()
    gan_model.train()
    gan_model.generate_and_visualize()
    print("Initial Design generated. How would you like to optimize it?")
