import torch
from torchvision import models, transforms, datasets
from torch import nn
from collections import OrderedDict
from PIL import Image
import numpy as np

model_choices = model_names = sorted(name for name in models.__dict__
    if name.islower() and name.startswith("v") and not name.endswith("n") and callable(models.__dict__[name]))

def build_model(arch, hidden_units):
    '''
    function to build model, returns model
    arch: string, model to call from torchvision
    hidden_units: list, number of units per hidden layer
    '''
    model = getattr(models, arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    layer_sizes = list(zip(hidden_units[:-1], hidden_units[1:]))
    
    modules = OrderedDict()
    
    for count, layer in enumerate(layer_sizes, 1):
        name = 'fc' + str(count)
        modules[name] = nn.Linear(layer[0], layer[1])
        if count == len(layer_sizes):
            modules['output'] = nn.LogSoftmax(dim=1)
        else:
            relu = 'relu' + str(count)
            modules[relu] = nn.ReLU()
            dropout = 'dropout' + str(count)
            modules[dropout] = nn.Dropout(p=0.5)
    
    # build classifier
    classifier = nn.Sequential(modules)
    
    model.classifier = classifier
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Pytorch tensor
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    
    # preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = transform(image)
    
    return image

def load_model(checkpoint):
    '''
    function to load model, returns model
    checkpoint: directory containing checkpoint to load
    '''
    checkpoint = torch.load(checkpoint)
    model = build_model(checkpoint["arch"], checkpoint["hidden_units"])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint["class_to_idx"]
    
    return model

def load_data(data_dir):
    '''
    function to load data, and apply the appropriate transforms
    returns trainloader, validloader, train_data.class_to_idx; data loaders for training and validation, 
    and dict containing class, indices pair
    data_dir: directory containing training data and validation data
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return trainloader, validloader, train_data.class_to_idx

def save_model(model, arch, hidden_units, save_dir, class_to_idx):
    '''
    function to save model to a .pth file
    model: model to be saved
    save_dir: directory to save the model
    '''
    model.class_to_idx = class_to_idx

    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
    }

    torch.save(checkpoint, save_dir + "/checkpoint.pth")
    
def main():
    '''
    debug function to test build_model
    '''
    arch = "vgg19"
    hidden_units = [25088, 4096, 4096, 102]
    model = build_model(arch, hidden_units)
    print(model)
    
if __name__ == "__main__":
    main()