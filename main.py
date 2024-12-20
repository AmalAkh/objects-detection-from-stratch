import torch
from torch.utils.data import DataLoader
from dataset import SimpleObjectDetectionDataset
from model import DetectionModel

dataset = SimpleObjectDetectionDataset("/Users/amalahmadinurov/Downloads/datasets")

dataloader = DataLoader(dataset, batch_size=1)

model = DetectionModel()
model.train(dataloader)
