import torch.nn as nn
import torch.optim as optim
from torchvision.models import VGG16_Weights, vgg16

vgg = vgg16(weights=VGG16_Weights.DEFAULT)
vgg.requires_grad_(False)

class DetectionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_fn = nn.MSELoss()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512,128),
            nn.ReLU(),
            nn.Linear(128, 4),  
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.classifier.parameters())
    def forward(self,x):
        features = vgg.features.forward(x)
        coords = self.classifier.forward(features)
        return coords
    def evaluate(self,dataloader):
        eval_loss = 0
        self.eval()
        for batch in iter(dataloader):
            
           
            x = batch[0].to("cpu")
            y = batch[1].to("cpu")
            predictions = self.forward(x)
            
            loss = self.loss_fn(predictions, y)
            eval_loss += loss.item()
        self.train(True)
        print("Loss:",eval_loss/len(dataloader.dataset))

         
        
        
    def fit(self,dataloader, epochs=10):
       
        for i in range(0,epochs):
            epochs_loss = 0
            for batch in iter(dataloader):
                
               
                x = batch[0].to("cpu")
                y = batch[1].to("cpu")
                predictions = self.forward(x)
                
                loss = self.loss_fn(predictions, y)
                epochs_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {i+1}, Loss: {epochs_loss/len(batch)}")
        

            
        





        