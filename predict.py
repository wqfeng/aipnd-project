# Predict flower name from an image with predict.py along with the probability of that name

# Basic usage: python predict.py input checkpoint
# Options:
# Return top K most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu

# import required modules
import argparse
import json
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from helper import process_image, load_model

# restructure predict file
def main():
    # define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help = "directory containing image")
    parser.add_argument("checkpoint", help = "checkpoint for loading pretrained model")
    parser.add_argument("--top_k", help = "Return top KK most likely classes", type = int, default = 1)
    parser.add_argument("--category_names", help = "Use a mapping of categories to real names", default = os.path.dirname(os.path.abspath(__file__)) + "/cat_to_name.json")
    parser.add_argument("--gpu", help = "Use GPU for inference", action = "store_true")
    args = parser.parse_args()

    # parse arguments
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    cuda = args.gpu
    
    # process image
    image = process_image(image_path)
    
    names, probs = predict(image, checkpoint, top_k, cat_to_name, cuda)
    
    # print name and probability
    for name, prob in zip(names, probs):
        print("{:<25} {:5.2f}%".format(name.title(), prob*100))
    
def predict(image, checkpoint, top_k, cat_to_name, cuda):
    '''
    function for predicting names and their respective probabilities, returns top_k names and their probabilities
    image: image tensor
    checkpoint: model to load and use for the prediction
    top_k: number of names, and probabilities to return
    cat_to_name: dictionary containing class and name pairs
    cuda: boolean, to use gpu or not to during testing
    '''
    
    # load checkpoint and build model
    model = load_model(checkpoint)

    # pass image to model
    # TODO: Implement the code to predict the class from an image file
    image = Variable(image.resize_(1, 3, 224, 224))
    model.eval()
    if cuda:
        model.cuda()
        image = image.cuda()
    else:
        model.cpu()
        image = image.cpu()
    logits = model.forward(image)
    ps = F.softmax(logits, dim=1)
    probs, indices = ps.topk(top_k)

    # convert from indices to classes to names
    idx_to_class = {y:x for x, y in model.class_to_idx.items()}

    probs = probs.cpu().data.numpy().squeeze()
    indices = indices.cpu().data.numpy().squeeze()
    
    # convert 0-d arrays to 1-d arrays when top_k produces only 1 result
    if top_k == 1:
        probs = np.array([probs])
        indices = np.array([indices])
    
    names = list(map(lambda i: cat_to_name[str(idx_to_class[i])], indices))
    
    return names, probs
        
if __name__ == '__main__':
    main()