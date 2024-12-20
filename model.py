import torch.nn as nn
import torch.optim as optim
from torchvision.models import VGG16_Weights, vgg16

vgg = vgg16(weights=VGG16_Weights.DEFAULT)


class DetectionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_fn = nn.MSELoss()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512,128),
            nn.ReLU(),
            nn.Linear(128,4)    
        )

        self.optimizer = optim.Adam(self.classifier.parameters)
    def forward(self,x):
        features = vgg.features.forward(x)
        coords = self.classifier.forward(features)
        return coords
    
    def train(self,dataloader, epochs=10):
        epochs_loss = 0
        for i in range(0,epochs):
            for batch in iter(dataloader):
                x = batch[0].to("cpu")
                y = batch[0].to("cpu")
                predictions = self.forward(x)
                
                loss = self.loss_fn(predictions, y)
                epochs_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print(epochs_loss/len(dataloader.dataset))






        