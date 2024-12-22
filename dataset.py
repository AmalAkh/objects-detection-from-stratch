from torchvision.io import decode_image, read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset
import torch
import math
import xml.etree.ElementTree as ET
import os
class SimpleObjectDetectionDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.images = []
        self.annotations = []

        for image_filename in os.listdir(os.path.join(path, "images")):
            
            if image_filename == ".DS_Store":
                continue
            image = Resize((224,224))(read_image(os.path.join(path, "images", image_filename)))
            image = image.float()
            
            self.images.append(image/255)
        


            
            tree = ET.parse(os.path.join(path, "annotations", image_filename.replace(".jpg", ".xml")))
            root = tree.getroot()
            

            for boxes in root.iter('object'):
                ymin, xmin, ymax, xmax = None, None, None, None

                ymin = int(boxes.find("bndbox/ymin").text)
                xmin = int(boxes.find("bndbox/xmin").text)
                ymax = int(boxes.find("bndbox/ymax").text)
                xmax = int(boxes.find("bndbox/xmax").text)

                center_x = int((xmax+xmin)/2)
                center_y = int((ymax+ymin)/2)
                
                row = int(math.floor(center_x/112))
                column = int(math.floor(center_y/112))

                coords = torch.zeros(4,4,4)
                coords[row][column] = torch.Tensor([xmin, ymin, xmax, ymax])
                print(coords[row][column])
                coords = coords.float()
                coords[row][column]/=224
              

                existance = torch.zeros(4,4,1)
                existance[row][column] = 1
                
                self.annotations.append((torch.flatten(coords), torch.flatten(existance)))
        print(f"Loaded {len(self.images)} files")

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return (self.images[index], self.annotations[index])

        
    