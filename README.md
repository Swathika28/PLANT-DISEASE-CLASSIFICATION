# PLANT-DISEASE-CLASSIFICATION
## INTRODUCTION:
Detection of plant disease has a crucial role in better understanding the economy of India in terms of agricultural productivity.Plant diseases can have significant negative impacts on agricultural productivity and ecosystems such as crop losses,increased pesticide use etc.Here We are building a model, which can classify between healthy and diseased crop leaves and also if the crop have any disease, predict which disease is it.We use ResNet, which have been one of the major breakthrough in computer vision.
ResNet, short for Residual Network is a specific type of neural network that was introduced in 2015 by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun in their paper “Deep Residual Learning for Image Recognition”.The ResNet models were extremely successful
The primary objective of this project is to develop a system that can accurately identify and classify diseases affecting plants based on images of their leaves. In cases of high yield loss and with subtle crops, the farmer or user can capture an image of the crop. Using the input image, the system will generate details about the plant disease. This project aims to detect plant diseases at an early stage; timely identification and treatment of plant diseases contribute to higher crop yields. Primarily, it assists terrace farmers and individuals engaged in farming as a hobby.

## PROBLEM STATEMENT:
Terrace farmers and individuals engaged in farming as a hobby frequently encounter challenges related to agricultural knowledge and resource access that may restrict their awareness of plant diseases. The amalgamation of limited formal education, the informal nature of hobby farming, resource constraints, and the absence of professional guidance contributes to a potential lack of awareness among terrace farmers and hobbyist individuals concerning plant diseases.This system helps farmers to gain knowlegde about the plant disease and its types.

## SCOPE OF THE PROJECT:
The primary objective of this project is to empower users to acquire more knowledge about plant diseases. We collect relevant input data from the user, such as a plant image. By processing and training the data, the system generates output corresponding to the specific disease predicted, matching the user-provided input image.

Currently, our project outputs plant disease names. However, we have the potential to enhance the user experience by expanding the output scope. This expansion could include detailed information about the cure for the identified plant diseases. This not only aids in disease identification but also enriches users' overall knowledge about plants, plant diseases, and their respective 

## METHODOLOGY:
The methodology employed in this project involves the utilization of ResNet (Residual Network) architecture, specifically ResNet99.2. In contrast to traditional neural networks, where each layer sequentially feeds into the next layer, ResNet employs residual blocks. In this architecture, each layer not only contributes to the subsequent layer but also has direct connections with layers approximately 2–3 hops away. This design is implemented to mitigate overfitting, a situation characterized by the validation loss ceasing to decrease and subsequently increasing, even as the training loss continues to decrease. Additionally, this approach helps address the vanishing gradient problem, enabling effective training of deep neural networks.
In the context of this project, ResNet plays a crucial role in enhancing accuracy, contributing to the effectiveness of the system.

## WORKING PROCESS
![image](WORKFLOW.png)

                                  

## PROGRAM
```python
import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import torch                    # Pytorch module 
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image           # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.utils import make_grid       # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images

%matplotlib inline
```

```python
data_dir = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)
nums = {}
for disease in diseases:
    nums[disease] = len(os.listdir(train_dir + '/' + disease))
```

```python
# converting the nums dictionary to pandas dataframe passing index as plant name and number of images as column
img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["no. of images"])
```

```python
# plotting number of images available for each disease
index = [n for n in range(38)]
plt.figure(figsize=(20, 5))
plt.bar(index, [n for n in nums.values()], width=0.3)
plt.xlabel('Plants/Diseases', fontsize=10)
plt.ylabel('No of images available', fontsize=10)
plt.xticks(index, diseases, fontsize=5, rotation=90)
plt.title('Images per each class of plant disease')
```

### Data Preparation
```python
# datasets for validation and training
train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor()) 
img, label = train[0]
print(img.shape, label)
random_seed = 7
batch_size = 32
torch.manual_seed(random_seed)
train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)
```

### Creation of Model
```python
# To get the device
def get_default_device():
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device 
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)
device = get_default_device()
device
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()  
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input
def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
#base class for the model
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))
```

### Final Architecture
```python
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out        
resnet = to_device(ResNet9(3, len(train.classes)), device) 
resnet
```

### Model Training
```python
@torch.no_grad()
def evaluate(resnet, val_loader):
    resnet.eval()
    outputs = [resnet.validation_step(batch) for batch in val_loader]
    return resnet.validation_epoch_end(outputs)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def fit_OneCycle(epochs, max_lr, resnet, train_loader, val_loader, weight_decay=0,
                grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(resnet.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        resnet.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = resnet.training_step(batch)
            train_losses.append(loss)
            loss.backward()   
            if grad_clip: 
                nn.utils.clip_grad_value_(resnet.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()
        result = evaluate(resnet, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        resnet.epoch_end(epoch, result)
        history.append(result)
%%time
history = [evaluate(resnet, valid_dl)]
history
epochs = 2
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```

### Graph plotting
```python
# PLOTING
def plot_accuracies(history):
    val_accuracies = [x['val_accuracy'] for x in history]
    epochs = range(1, len(val_accuracies) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accuracies, 'b', marker='o', label='Validation Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', marker='x', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    batch_nums = range(1, len(lrs) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(batch_nums, lrs, 'g', marker='x', label='Learning Rate')
    plt.title('Learning Rate Over Batches')
    plt.xlabel('Batch Number')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.legend()
    plt.show()
plot_accuracies(history)
plot_losses(history)
plot_lrs(history)
test_dir = "../input/new-plant-diseases-dataset/test"
test = ImageFolder(test_dir, transform=transforms.ToTensor())
test_images = sorted(os.listdir(test_dir + '/test')) # since images in test folder are in alphabetical order
test_images
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return train.classes[preds[0].item()]
# predicting first image
img, label = test[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[0], ', Predicted:', predict_image(img, resnet))
Saving the model
```

### Saving the Model
```python
PATH = './model.pth'
torch.save(resnet, PATH)
```

## INPUT


## OUTPUT


## RESULT

