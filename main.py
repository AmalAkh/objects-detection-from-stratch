import torch
from torch.utils.data import DataLoader
from dataset import SimpleObjectDetectionDataset
from model import DetectionModel

train_dataset = SimpleObjectDetectionDataset("simple-object-detection-dataset/train")
test_dataset = SimpleObjectDetectionDataset("simple-object-detection-dataset/test")



train_dataloader = DataLoader(train_dataset, batch_size=10)
test_dataloader = DataLoader(test_dataset, batch_size=1)


model = DetectionModel()
model.fit(train_dataloader, epochs=40)
torch.save(model.state_dict(), "./model2")
model.evaluate(test_dataloader)




