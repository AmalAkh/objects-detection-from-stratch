import torch
from torch.utils.data import DataLoader
from dataset import SimpleObjectDetectionDataset
from model import DetectionModel
from torchvision.io import decode_image, read_image
from torchvision.transforms import Resize
import cv2
import numpy as np
train_dataset = SimpleObjectDetectionDataset("simple-object-detection-dataset/test")
train_dataloader = DataLoader(train_dataset, batch_size=10)


model = DetectionModel()
model.load_state_dict(torch.load("./model"))

model.eval()
model.evaluate(train_dataloader)
'''
image = Resize((224,224))(read_image("/Users/amalahmadinurov/Downloads/datasets/images/a (1).jpg"))
image = image.float()/255
model_input = torch.stack([image])

result = model.forward(model_input)
result = (result*224).int()
print(result)
img = cv2.imread("/Users/amalahmadinurov/Downloads/datasets/images/a (1).jpg")

img = cv2.rectangle(img, result.numpy()[0], (0, 255, 0), 2)
cv2.imshow("",img)
cv2.waitKey(0)
'''