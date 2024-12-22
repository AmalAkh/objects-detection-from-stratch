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
model.load_state_dict(torch.load("./model2"))

model.eval()
#model.evaluate(train_dataloader)
img_path = "/Users/amalahmadinurov/Desktop/simple-objects-detector/simple-object-detection-dataset/test/images/a (110).jpg"
image = Resize((224,224))(read_image(img_path))
image = image.float()/255
model_input = torch.stack([image])

result = model.forward(model_input)
result = torch.reshape(result*224, (4,4,4)).int()
print(result)



img = cv2.imread(img_path)


for block in result.numpy():
    for box in block:
        img = cv2.rectangle(img, box, (0, 255, 0), 2)


cv2.imshow("",img)
cv2.waitKey(0)
'''
img = cv2.imread("/Users/amalahmadinurov/Downloads/datasets/images/a (3).jpg")

img = cv2.rectangle(img, result[0][0].numpy(), (0, 255, 0), 2)
img = cv2.rectangle(img, result[0][1].numpy(), (0, 255, 0), 2)
cv2.imshow("",img)
cv2.waitKey(0)
'''