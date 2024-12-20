from torchvision.io import decode_image, read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os
class SimpleObjectDetectionDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.images = []
        self.annotations = []

        for image_filename in os.listdir(os.path.join(path, "images")):
            print(os.path.join(path, "images", image_filename))
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

                list_with_single_boxes = [xmin, ymin, xmax, ymax]
                self.annotations.append(list_with_single_boxes)

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return (self.images[index], self.annotations[index])

        
    