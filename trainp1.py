from model_grading import MultiTaskModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

model = MultiTaskModel(5, 4, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = (
            F.cross_entropy(outputs["clarity"], labels["clarity"]) +
            F.cross_entropy(outputs["color"], labels["color"]) +
            F.cross_entropy(outputs["cut"], labels["cut"]) +
            F.mse_loss(outputs["carat"].squeeze(), labels["carat"])
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
