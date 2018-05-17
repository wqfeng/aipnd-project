# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu

# import required modules
import os
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from helper import build_model, model_choices, load_data, save_model

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help = "directory containing training data")
    parser.add_argument("--save_dir", help = "directory to save checkpoint", default = os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--arch", help = "choose architecture", choices = model_choices, default = "vgg19")
    parser.add_argument("--learning_rate", help = "set learning rate", type = float, default = 0.001)
    parser.add_argument("--hidden_units", help = "set hidden units", default = [25088, 4096, 4096, 102])
    parser.add_argument("--epochs", help = "set number of epochs to train for", type = int, default = 5)
    parser.add_argument("--gpu", help = "use GPU for training", action = "store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    lr = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    cuda = args.gpu
    
    # load data
    trainloader, validloader, class_to_idx = load_data(data_dir)
    
    # TODO: Build and train your network
    model = build_model(arch, hidden_units)
    
    # TODO: Train a model with a pre-trained network
    model = train(model, epochs, lr, cuda, trainloader, validloader)
    
    # TODO: Save the checkpoint
    save_model(model, arch, hidden_units, save_dir, class_to_idx)
    
def train(model, epochs, lr, cuda, trainloader, validloader):
    '''
    function for training model, returns trained model
    model: model to be trained
    epochs: number of epochs to train for
    lr: learning rate to apply for each pass
    cuda: boolean, to use gpu or not to during testing
    trainloader: data loader for training data
    validloader: data loader for validation data
    '''
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    print_every = 40
    steps = 0
    for e in range(epochs):
        model.train()

        running_loss = 0
        for inputs, labels in iter(trainloader):
            steps += 1

            # convert images and labels to use autograd
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            # forward pass
            if cuda:
                model.cuda()
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                model.cpu()
                inputs, labels = inputs.cpu(), labels.cpu()

            outputs = model.forward(inputs)

            # backward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                accuracy = 0
                valid_loss = 0
                for ii, (inputs, labels) in enumerate(validloader):

                    # Set volatile to True so we don't save the history
                    inputs = Variable(inputs, volatile=True)
                    labels = Variable(labels, volatile=True)

                    # forward pass
                    if cuda:
                        model.cuda()
                        inputs, labels = inputs.cuda(), labels.cuda()
                    else:
                        model.cpu()
                        inputs, labels = inputs.cpu(), labels.cpu()


                    output = model.forward(inputs)
                    valid_loss += criterion(output, labels).data[0]

                    ## Calculating the accuracy 
                    # Model's output is log-softmax, take exponential to get the probabilities
                    ps = torch.exp(output).data
                    # Class with highest probability is our predicted class, compare with true label
                    equality = (labels.data == ps.max(1)[1])
                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure dropout is on for training
                model.train()

    print("Completed training")
        
    # move model back to cpu
    model.cpu()
        
    return model

if __name__ == '__main__':
    main()