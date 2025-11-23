import json
import os
from PIL import Image
import torch
from torchvision import transforms


class SeqsDataSet:
    def __init__(self, json_path, images_path, labels_path, transform=None):
        with open(json_path, 'r') as f:
            self.seqs = json.load(f)
        self.images_path = images_path
        self.labels_path = labels_path
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(), 
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        
        seq_files = self.seqs[idx]
        seq_images = []
        seq_labels = []
        for seq_file in seq_files:
            image_path = os.path.join(self.images_path, seq_file[0:6], seq_file.replace('.png', '_bright_img.png'))
            label_path = os.path.join(self.labels_path, seq_file[0:6], seq_file)   
            image, label = Image.open(image_path), Image.open(label_path)
            image, label = self.transform(image), self.transform(label)
            seq_images.append(image)
            seq_labels.append(label)
        return torch.stack(seq_images), torch.stack(seq_labels)

class NoSeqsDataSet:
    def __init__(self, json_path, images_path, labels_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.images_path = images_path
        self.labels_path = labels_path
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(), 
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        file = self.data[idx]
        
        image_path = os.path.join(self.images_path, file[0:6], file.replace('.png', '_bright_img.png'))
        label_path = os.path.join(self.labels_path, file[0:6], file)   
        image, label = Image.open(image_path), Image.open(label_path)
        image, label = self.transform(image), self.transform(label)
            
        return image, label