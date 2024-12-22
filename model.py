import torch.nn as nn
import torch.optim as optim
from torchvision.models import VGG16_Weights, vgg16
import torch 
vgg = vgg16(weights=VGG16_Weights.DEFAULT)
vgg.requires_grad_(False)

class DetectionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_fn = nn.MSELoss()

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512,256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 4*4*4),  
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512,128),
            nn.ReLU(),
            nn.Linear(128,16),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters())
    def forward(self,x):
        features = vgg.features.forward(x)
        coords = self.regressor.forward(features)
        object_exists = self.classifier.forward(features)

        return (coords, object_exists)
    def evaluate(self,dataloader):
        reg_eval_loss = 0
        class_eval_loss = 0
        self.eval()
        for batch in iter(dataloader):
            
           
            x = batch[0].to("cpu")
            y_reg = batch[1][0].to("cpu")
            y_class = batch[1][1].to("cpu")
            predictions = self.forward(x)
            
            reg_loss = self.loss_fn(predictions[0], y_reg)
            class_loss = self.loss_fn(predictions[0], y_class)

            reg_eval_loss += reg_loss.item()
            class_eval_loss += class_loss.item()

        self.train(True)
        print(f"Regression Loss:{reg_eval_loss/len(dataloader.dataset)} Classification Loss: {class_eval_loss/len(dataloader.dataset)}")

         
        
        
    def fit(self,dataloader, epochs=10):
       
        for i in range(0,epochs):
            epoch_reg_loss = 0
            epoch_class_loss = 0
            for batch in iter(dataloader):
                
               
                x = batch[0].to("cpu")
                y_reg = batch[1][0].to("cpu")
                y_class = batch[1][1].to("cpu")


                predictions = self.forward(x)
                
                regression_loss = self.loss_fn(predictions[0], y_reg)
                epoch_reg_loss += regression_loss.item()
                
                self.optimizer.zero_grad()
                regression_loss.backward()
                self.optimizer.step()

                classifier_loss = self.loss_fn(predictions[1], y_class)
                epoch_class_loss += classifier_loss.item()
                
                self.optimizer.zero_grad()
                classifier_loss.backward()
                self.optimizer.step()

            print(f"Regression Loss:{epoch_reg_loss/len(batch[0])} Classification Loss: {epoch_class_loss/len(batch[0])}")
            
        

            
        





        