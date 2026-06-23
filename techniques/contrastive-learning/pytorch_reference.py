
# ============================================================
# PYTORCH IMPLEMENTATION (Referensi untuk Produksi)
# ============================================================
# Jalankan setelah: pip install torch torchvision
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torchvision

# 1. Model SimCLR
class SimCLR(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        n_features = backbone.fc.in_features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return torch.flatten(h, 1)

    def forward(self, x):
        h = self.encode(x)
        z = self.projection_head(h)
        return h, z

# 2. NT-Xent Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.T = temperature

    def forward(self, z_i, z_j):
        N = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)               # (2N, D)
        sim = torch.mm(z, z.T) / self.T                 # (2N, 2N)
        mask = torch.eye(2*N, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))
        target = torch.cat([
            torch.arange(N, 2*N),
            torch.arange(0, N)
        ]).to(z.device)
        return F.cross_entropy(sim, target)

# 3. Augmentation Transform
class ContrastiveTransform:
    def __init__(self, size=96):
        self.t = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __call__(self, x):
        return self.t(x), self.t(x)

# 4. Training Loop
def train_simclr(model, loader, epochs=200, lr=3e-4, temperature=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = NTXentLoss(temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (x_i, x_j), _ in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
    return model

# 5. Dataset STL-10 + Pretrain
if __name__ == "__main__":
    dataset = torchvision.datasets.STL10(
        root="./data", split="unlabeled",
        download=True,
        transform=ContrastiveTransform(96)
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        num_workers=4, drop_last=True)
    model = SimCLR(base_model="resnet18", out_dim=128)
    model = train_simclr(model, loader, epochs=200)
    torch.save(model.state_dict(), "simclr_pretrained.pt")
