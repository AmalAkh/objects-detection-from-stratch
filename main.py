import torch
from torch.utils.data import DataLoader
from dataset import SimpleObjectDetectionDataset
from model import DetectionModel

train_dataset = SimpleObjectDetectionDataset("simple-object-detection-dataset/train")

train_dataloader = DataLoader(train_dataset, batch_size=10)

model = DetectionModel()
model.fit(train_dataloader)

torch.save(model.state_dict(), "./model")
