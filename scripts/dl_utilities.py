# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The MIT License (MIT)
# Copyright (c) 2023 Thomas DUMAZERT
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# ----------------------------------------------
# | script: dl_utilities.py                    |
# | author: Thomas DUMAZERT                    |
# | creation: 03/23/2023                       |
# | last modified: 03/23/2023                  |
# ----------------------------------------------

# This module is intended to provided utility functions to train Deep Learning 
# models using Pytorch.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# External libraries

import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constants

# Device available to train the models
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = '../data'
IMAGES_DIR = f'{DATA_DIR}/images'
MODELS_SAVE_DIR = f'{DATA_DIR}/models'
REPORTS_DIR = f'{DATA_DIR}/models_results'
CLASS_REPORTS_DIR = f'{REPORTS_DIR}/classification_reports'
GRADCAM_DIR = f'{REPORTS_DIR}/gradCAMs'

# Number of metrics registered during models training
# [train_loss, val_loss, train_accuracy, val_accuracy]
N_METRICS = 4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def plot_history(history):
    """
    Plot the loss and the metrics evolution over epochs of a model training 
    process.
    Parameter:
        - 'history': dictionary. Contains the loss and the metrics of computed 
                     during the training process.
    Return: None
    """

    train_acc = history['accuracy']
    val_acc = history['val_accuracy']

    train_loss = history['loss']
    val_loss =  history['val_loss']

    n_epochs = len(train_acc)
    epochs = range(1, n_epochs+1)
    
    _, ax = plt.subplots(ncols=2, figsize=(20, 10))
    ax[0].plot(epochs, train_loss, color='gray', label='Train')
    ax[0].plot(epochs, val_loss, color='orange', label='Validation')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(epochs, train_acc, color='gray', label='Train')
    ax[1].plot(epochs, val_acc, color='orange', label='Validation')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.show()

def save_model(dir, model, name, history=None):
    """
    Save a model (its architecture and their parameters) and its training 
    history in a dedicated directory.
    Parameters:
        - 'dir': string. Path to the directory where to store the model ;
        - 'model': a trained model ;
        - 'name': string. The name of the directory to create to store the 
                  model ;
        - 'history': optionnal, dictionary. Contains the loss and the metrics 
                     of computed during the training process.
    Return: the path to the created directory.
    """

    # Create the directory to store the model and its history
    target_dir = Path(f'{dir}/{name}_001')

    # If the directory already exists, create a new one with incremented last 
    # digits
    if target_dir.exists():
        
        def get_dir_name(d):
            """
            Extracts the base name of a directory from a name in the form:
            ppp/nnn_ddd
            Where:
                - 'ppp': is the path to the directory ;
                - 'nnn': is the directory base name ;
                - 'ddd': is the unique id of the directory.
            Parameter:
                - 'd': string. The directory full path.
            Return: the directory base name.
            """

            return '_'.join(split_path(d)[-1].split('_')[:-1])
        
        base_dir = get_dir_name(str(target_dir))

        # List all the directories which have the same name
        list_dir = [split_path(d)[-1] for d in os.listdir(dir) if get_dir_name(d) == base_dir]

        # Extract their number
        num_dir = [int(d.split('_')[-1]) for d in list_dir]

        # Find the next number to create
        next_num = max(num_dir)+1
        target_dir = Path(f'{dir}/{name}_{next_num:03d}')

    os.mkdir(target_dir)

    model_file = Path(f'{target_dir}/model.h5')
    model.save(model_file)

    if history:
        # Convert the history dictionary to a DataFrame to save it in csv
        history_name = Path(f'{target_dir}/history.csv')
        pd.DataFrame(history).to_csv(history_name, index=False)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=30, device=DEVICE, is_inception=False):
    """
    Function used to train pretrained based model in Pytorch
    Parameters:
        - 'model': torch.Model object. The model to train ;
        - 'dataloaders': torch.utils.data.DataLoader. The Dataloader providing
                         train and validation data ;
        - 'criterion': function. The loss function to use during the 
                       training ;
        - 'optimizer': function. The optimization function to use during the
                       training ;
        - 'num_epochs': int, optional. The number of epochs. Default to 30 ;
        - 'device': string, optional. The device (cuda or cpu) on which to 
                    train the model. Default to the device detected by the 
                    script ;
        - 'is_inception': boolean. Flag indicating if the model is Inception 
                          v3. Default to False.
    Return: trained model and training history.
    """

    model = model.to(device)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and (phase == 'train'):
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = float(running_loss / len(dataloaders[phase].dataset))
            epoch_acc = float(running_corrects.double() / len(dataloaders[phase].dataset))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, [train_loss_history, val_loss_history, train_acc_history, val_acc_history]

def set_parameter_requires_grad(model, trainable):
    """
    Function used to change the trainability of the layers of a model.
    Parameters:
        - 'model': torch.Model object. The model to modify ;
        - 'trainable': boolean. Whether the model should trainable or not.
    Return: the modified model.
    """
    for param in model.parameters():
        param.requires_grad = trainable
    
    return model

def initialize_model(model_name, num_classes, trainable=False):
    """
    Function creating a torch.Model object.
    Parameters:
        - 'model_name': string. The name of the model to create ;
        - 'num_classes': int. The  number of classes corresponding to the 
                         number of node of the classification layer ;
        - 'trainable': boolean, optional. Whether the model will be used as a 
                       feature extractor (False) or not (True). Default to 
                       False.
    Return: a torch.Model object and the input size of the images accepted by 
            the model.
    """

    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        # Resnet18
        model_ft = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model_ft = set_parameter_requires_grad(model_ft, trainable)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        # Alexnet
        model_ft = models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
        model_ft = set_parameter_requires_grad(model_ft, trainable)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        # VGG11_bn
        model_ft = models.vgg11_bn(weights=torchvision.models.VGG11_BN_Weights.DEFAULT)
        model_ft = set_parameter_requires_grad(model_ft, trainable)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        # Squeezenet
        model_ft = models.squeezenet1_0(weights=torchvision.models.SqueezeNet1_0_Weights.DEFAULT)
        model_ft = set_parameter_requires_grad(model_ft, trainable)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        # Densenet
        model_ft = models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
        model_ft = set_parameter_requires_grad(model_ft, trainable)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        # Inception v3
        # Be careful, expects (299,299) sized images and has auxiliary output
        model_ft = models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
        model_ft = set_parameter_requires_grad(model_ft, trainable)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def create_dataloader(dir, data_transforms, batch_size=32):
    """
    Function used to create a Pytorch image dataset and dataloader.
    Parameters:
        - 'dir': string or pathlib.Path object. The directory where to find 
                 the images ;
        - 'data_transforms': a torchvision.transform, or 
                             torchvision.tansforms.Compose object. The 
                             transformations to apply ;
        - 'batch_size': int, optional. The number of images by batch to 
                        provide to the model. Default to 32.
    Return: the dataloader and the image dataset.
    """

    # Create datasets
    images_dataset = datasets.ImageFolder(dir, data_transforms)

    # Create dataloaders
    dataloader = torch.utils.data.DataLoader(images_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader, images_dataset

def create_dataloaders(root_dir, input_size, batch_size=32):
    """
    Function used to create the train, test, and validation dataloaders and 
    datasets.
    Parameters:
        - 'root_dir': a sting or pathlib.Path object. The directory where to 
                      find the sets ;
        - 'input_size': int. The size to which to resize the images ;
        - 'batch_size': int, optional. The number of images by batch to 
                        provide to the model. Default to 32.
    Return: a dictionnary of dataloaders and a dictionnary of datasets.
    """

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataloaders_dict = {}
    datasets_dict = {}

    for x in ['train', 'val', 'test']:
        dataloaders_dict[x], datasets_dict[x] = create_dataloader(f'{root_dir}/{x}', data_transforms[x], batch_size)

    return dataloaders_dict, datasets_dict

def save_hists(hists, path):
    """
    Function saving histories to the specified path.
    Parameters:
        - 'hists': a list of lists. The histories to save ;
        - 'path': a string or patlib.Path object. The path to the file in 
                  which to save the histories.
    Return: None.
    """

    with open(path,'w') as tfile:
        for hist in hists:
            tfile.write(', '.join(map(str, hist)))
            tfile.write('\n')

def load_hists(path):
    """
    Function loading the training histories from the specified file.
    Parameter:
        - 'path': a string or patlib.Path object. The path to the file in 
                  which the histories are saved.
    Return: a list of lists of the histories.
    """
    
    hists = []
    with open(path, 'r') as rfile:
        hist = rfile.readlines()
    
    hists = [[float(n) for n in h.split(', ')] for h in hist]

    return hists

def hists_to_dict(l_hists, ds_names):
    """
    Function converting a list of lists containing training histories to a 
    dictionnary.
    Parameters:
        - 'l_hists': a list of lists. The training histories ;
        - 'ds_dirs': a list of strings. The list of the datasets names.
    Return: a dictionnary of the histories organized by dataset by steps and 
            metrics.
    """

    hists = {}

    for i, ds in enumerate(ds_names):
        train_loss, val_loss = l_hists[i*N_METRICS+0], l_hists[i*N_METRICS+1]
        train_accuracy, val_accuracy = l_hists[i*N_METRICS+2], l_hists[i*N_METRICS+3]
        hists[ds] = {
            'loss': {'train': train_loss, 'val': val_loss},
            'accuracy': {'train': train_accuracy, 'val': val_accuracy}
        }
        
    return hists

def load_model(model_name, save_path, num_classes, trainable=False):
    """
    Function creating a pretrained model and loading its previously trained 
    weigths.
    Parameters:
        - 'model_name': a string. The name of the model to create ;
        - 'save_path': a string or pathlib.Path object. The path to the file 
                       where the pretrained weights are saved ;
        - 'num_classes': int. The  number of classes corresponding to the 
                         number of node of the classification layer ;
        - 'trainable': boolean, optional. Whether the model will be used as a 
                       feature extractor (False) or not (True). Default to 
                       False.
    Return: a torch.Model object with its pretained weigths and the input size 
            of the images accepted by the model.
    """

    model, input_size = initialize_model(model_name, num_classes, trainable=trainable)
    model.load_state_dict(torch.load(save_path))
    return model, input_size

def plot_report(cr, cm, categories, report_name=None, cmap='Blues', save_path=None, display_report=True):
    """
    Function plotting the classification report and the confusion matrix as 
    heatmaps, on the same figure.
    Parameters:
        - 'cr': a matrix. The classification report to plot ;
        - 'cm': a matrix. The confusion matrix to plot ;
        - 'categories': a list. The list of the categories' names ;
        - 'report_name': a string, optional. The title of the report displayed 
                         as the title of the figure. If None, the figure won't 
                         have title. Default to None ;
        - 'cmap': a colormap, optional. The color map to use. Default to 
                  'Blues' ;
        - 'save_path': a string or pathlib.Path object, optional. The path to 
                       the file to save the figure. If None, the figure won't 
                       be saved. Default to None ;
        - 'display_report': a boolean, optional. Whether to display or not the 
                            report. Default to True.
    Return: None.
    """

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))

    sns.heatmap(cm, xticklabels=categories, yticklabels=categories, cmap=cmap, fmt='d', ax=ax[0], annot=True, annot_kws={'size': 9})
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, cmap=cmap, fmt='.2f', ax=ax[1], annot=True, annot_kws={'size': 9})

    if report_name is not None:
        fig.suptitle(report_name)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    
    if display_report:
        plt.show();

def model_report(model, test_dataloader, test_dataset, model_name, ds_name, save_dir=None, device=DEVICE, display_report=True):
    """
    Function creating the performance report of a trained model.
    Parameters:
        - 'model': a torch.Model object. The model for which to create a 
                   performance report ;
        - 'test_dataloader': a torch.utils.data.Dataloader object. The 
                             dataloader supplying test images to the model ;
        - 'test_dataset': a torch.utils.data.Dataset object. The dataset 
                          containing the test set information ;
        - 'model_name': a string. The name of the model for which to create a 
                        performance report ;
        - 'ds_name': a string. The name of the dataset used to train the 
                     model ;
        - 'save_dir': a string or pathlib.Path object, optional. The directory 
                      in which to save the report. If None, the report won't 
                      be saved. Default to None ;
        - 'device': string, optional. The device (cuda or cpu) to use to 
                    predict labels. Default to the device detected by the 
                    script ;
        - 'display_report': a boolean, optional. Whether to display or not the 
                            report. Default to True.
    Return: None.
    """

    categories = test_dataset.class_to_idx

    y_test, y_pred = predict(model, test_dataloader, device)

    # Plot the report
    cr = classification_report(y_test, y_pred, target_names=categories, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report_name = f'{model_name} - {ds_name}'

    if save_dir is None:
        save_path = None
    else:
        save_path = f'{save_dir}/{model_name}-{ds_name.replace("/", "_")}.png'
    plot_report(cr, cm, categories, report_name, save_path=save_path, display_report=display_report)

def get_target_layers(model_name, model):
    """
    Function returning the layer to target for a gradCAM analysis.
    Parameters:
        - 'model_name': a string. The name of the model which will be 
                        analyzed ;
        - 'model': a torch.Model object. The model which will be analyzed.
    Return: a list containing the targeted layer.
    """

    if model_name == "resnet":
        # Resnet18
        return [model.layer4[-1]]

    elif model_name == "alexnet":
        # Alexnet
        return [model.features[-1]]

    elif model_name == "vgg":
        # VGG11_bn
        return [model.features[-1]]

    elif model_name == "squeezenet":
        # Squeezenet
        return [model.features[-2]]

    elif model_name == "densenet":
        # Densenet
        return [model.features[-1]]

    elif model_name == "inception":
        # Inception v3
        return [model.Mixed_7c]

    else:
        print("Invalid model name, exiting...")
        exit()

def normalize_image(img):
    """
    Function normalizing the pixels values of an image.
    Parameter:
        - 'img': a numpy array. The image to normalize.
    Return: a numpy array containing the normalized image.
    """

    divider = (np.max(img) - np.min(img))
    if divider == 0: divider = 1

    return (img - np.min(img)) / divider

def gradCAM(model, target_layers, img):
    """
    Function performing a gradCAM analysis.
    Parameters:
        - 'model': a torch.Model object. The model to analyze ;
        - 'target_layers': a list. The list of the layers to consider for the 
                           analysis ;
        - 'img': a numpy array. The image to use to perform the analysis.
    Return: a numpy array of the gradCAM heatmap surimpressded on the original 
            image.
    """

    # Max min normalization
    rgb_img = normalize_image(img)

    # Create an input tensor image for the model
    input_tensor = torch.tensor(rgb_img).unsqueeze(0).float()
    print(input_tensor.shape)
    model = model.to('cpu')

    with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)

    grayscale_cam = grayscale_cam[0, :]
    print(grayscale_cam.shape)

    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

def plot_images_couples(imgs1, imgs2, ncols=3, labels=None, title=None, save_path=None, display_img=True):
    """
    Function displaying couples of images, side by side on the same figure.
    Parameters:
        - 'imgs1': a list. The list of image which will on the left side of 
                   the couples ;
        - 'imgs2': a list. The list of image which will on the right side of 
                   the couples ;
        - 'ncols': int, optional. The number of columns in the figure. Default 
                   to 3 ;
        - 'labels': a list. The list of the couples titles. If None, couples 
                    woon't have a title. Default to None ;
        - 'title': a string, optional. The title of the figure. If None figure 
                   won't have a title. Default to None ;
        - 'display_img': a boolean, optional. Whether to display or not the 
                         resulting figure. Default to True.
    Return: None.
    """

    if len(imgs1) != len(imgs2):
        print('imgs1 and imgs2 are note the same size, exiting...')
        exit()
    else:
        n_imgs = len(imgs1)

    nrows = int(np.ceil(n_imgs / ncols))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 8))

    for i in range(ncols*nrows):
        row, col = i % ncols, i // ncols
        if i < n_imgs:
            if imgs1[i] is None or imgs2[i] is None: pass
            else:
                composed_img = np.concatenate((imgs1[i], imgs2[i]), axis=1)
                axs[col, row].imshow(composed_img)
            axs[col, row].set_xticks([])
            axs[col, row].set_yticks([])
            if labels is None: pass
            else: axs[col, row].set_title(labels[i])
        else:
            axs[col, row].set_axis_off()

    if title is None: pass
    else: fig.suptitle(title)
    
    if save_path is None: pass
    else: fig.savefig(save_path, bbox_inches='tight')
    
    if display_img: plt.show();

def proceed_gradCAMs(model, test_dataloader, test_dataset, model_name, ds_name, device=DEVICE, save_dir=None, display_img=True):  
    """
    Function performings gradCAM analysis on all the classes of the given 
    dataset.
    Parameters:
        - 'model': a torch.Model object. The model to analyze ;
        - 'test_dataloader': a torch.utils.data.Dataloader object. The 
                             dataloader supplying test images to the model ;
        - 'test_dataset': a torch.utils.data.Dataset object. The dataset 
                          containing the test set information ;
        - 'model_name': a string. The name of the model for which to create a 
                        performance report ;
        - 'ds_name': a string. The name of the dataset used to train the 
                     model ;
        - 'device': string, optional. The device (cuda or cpu) to use to 
                    predict labels. Default to the device detected by the 
                    script ;
        - 'save_dir': a string or pathlib.Path object, optional. The directory 
                      in which to save the report. If None, the report won't 
                      be saved. Default to None ;
        - 'display_img': a boolean, optional. Whether to display or not the 
                         resulting figure. Default to True.
    Return: None.
    """

    class_to_idx = test_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_idx = [i for i in idx_to_class.keys()]

    # Get real labels
    y_test = np.array(test_dataset.targets)

    _, y_pred = predict(model, test_dataloader, device)


    # Training must be enabled to process gradCAM
    model = set_parameter_requires_grad(model, trainable=True)

    target_layers = get_target_layers(model_name, model)

    # Select randomly 1 correctly labeled image per category
    correct_id = []
    
    for cat in class_idx:
        condition = ((y_pred == y_test) & (y_test == cat))
        if condition.sum() > 0:
            rnd_id = np.random.choice(np.argwhere(condition). reshape(-1,))
            correct_id.append(int(rnd_id))
        else:
            correct_id.append(None)
    
    images = [None if i is None else Image.open(test_dataset.imgs[i][0]).convert('RGB') for i in correct_id]

    # Get gradCAM for each images
    gradCAM_images = []
    for img in images:
        if img is None: gradCAM_images.append(None)
        else: gradCAM_images.append(gradCAM(model, target_layers, img))

    # Display the original images and the modified images
    images_labels = [i for i in idx_to_class.values()]
    save_path = f'{save_dir}/{model_name}-{ds_name.replace("/", "_")}-ok-gradCAM.png' if save_dir else None
    plot_images_couples(images, gradCAM_images, labels=images_labels, title=f'{model_name} - {ds_name}: correctly classed images', save_path=save_path, display_img=display_img)

    # Select random 1 wrongly labeled image per category
    wrong_id = []

    for cat in class_idx:
        condition = ((y_pred != y_test) & (y_test == cat))
        if condition.sum() > 0:
            rnd_id = np.random.choice(np.argwhere(condition). reshape(-1,))
            wrong_id.append(int(rnd_id))
        else:
            wrong_id.append(None)

    images = [None if i is None else Image.open(test_dataset.imgs[i][0]).convert('RGB') for i in wrong_id]

    # Get gradCAM for each images
    gradCAM_images = []
    for img in images:
        if img is None: gradCAM_images.append(None)
        else: gradCAM_images.append(gradCAM(model, target_layers, img))
    
    # Display the original images and the modified images
    real_labels = [i for i in idx_to_class.values()]
    predicted_labels = [None if i is None else idx_to_class[y_pred[i]] for i in wrong_id]
    images_labels = [f'{r} - {p}' for r, p in zip(real_labels, predicted_labels)]
    save_path = f'{save_dir}/{model_name}-{ds_name.replace("/", "_")}-nok-gradCAM.png' if save_dir else None
    plot_images_couples(images, gradCAM_images, labels=images_labels, title=f'{model_name} - {ds_name}: wrongly classed images', save_path=save_path, display_img=display_img)

def discriminate(models_names, ds_dirs, num_classes, num_epochs=30, batch_size=32, class_report=True, gradCAM_analysis=False, device=DEVICE, display_reports=True, 
                 models_save_dir=MODELS_SAVE_DIR, images_dir=IMAGES_DIR, class_reports_dir=CLASS_REPORTS_DIR, gradCAM_dir=GRADCAM_DIR):
    """
    Function taining couples model/dataset, performing performance and 
    gradCAM analysis and collecting training histories per model.
    Parameters:
        - 'model_names': a list. The list of the models names to train ;
        - 'ds_dirs': a list. The list of the datasets names to use for 
                     training ;
        - 'num_classes': int. The number of class to identify ;
        - 'num_epochs': int, optional. The number of epochs to train the 
                        models. Default to 30 ;
        - 'batch_size': int, optional. The number of image to supply to the 
                        model per batch. Deafult to 32 ;
        - 'class_report': boolean, optional. Whether to compute classification 
                          report or not. Default to True ;
        - 'gradCAM_analysis': boolean, optional. Whether or not to proceed 
                              gradCAM analysis or not. Default to False ;
        - 'device': string, optional. The device (cuda or cpu) to use to 
                    train the model. Default to the device detected by the 
                    script ;
        - 'display_reports': a boolean. Whether to display the performance and 
                             gradCAM reports or not. Default to True ;
        - 'models_save_dir': a string or pathlib.Path object, optional. The 
                             directory where to save the trained models. 
                             Default to MODEL_SAVE_DIR ;
        - 'images_dir': a string or pathlib.Path object, optional. The 
                        directory where to find the images. Default to 
                        IMAGES_DIR ;
        - 'class_reports_dir': a string or pathlib.Path object, optional. The 
                               directory where to save the performance reports. 
                               Default to CLASS_REPORTS_DIR ;
        - 'gradCAM_dir': a string or pathlib.Path object, optional. The 
                         directory where to save the gradCAMs analysis. 
                         Default to GRADCAM_DIR.
    Retturn: None
    """
    
    for model_name in models_names:
        print('\n' + '- ' * 10)
        print(f'Model {model_name}')
        hists = []

        # Create model directory
        model_dir = f'{models_save_dir}/{model_name}'
        if not Path(model_dir).exists(): os.mkdir(model_dir)

        for ds_dir in ds_dirs:
            print('~ ' * 10)
            print(f'Dataset: {ds_dir}')
            data_dir = f'{images_dir}/{ds_dir}'

            print('Model initialization ...', end=' ')
            # Initialize the model
            model, input_size = initialize_model(model_name, num_classes, False)
            print(f'done')

            print('Dataloader creation ...', end=' ')
            # Create training and validation dataloaders
            dataloaders_dict, datasets_dict = create_dataloaders(data_dir, input_size, batch_size)
            print('done')

            # Observe that all parameters are being optimized
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            print('Training ...')
            # Train and evaluate
            model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"), device=device)

            # Save the model
            ds_name = ds_dir.replace("/", "_")
            save_name = f'{model_name}-{ds_name}.pth'
            save_path = f'{model_dir}/{save_name}'

            torch.save(model.state_dict(), save_path)
            print(f'Saved PyTorch Model State to: {save_path}')

            if class_report:
                print('Classification report creation ...', end=' ')
                # Evaluate the model on the tests set and save the evaluation report
                model_report(model, dataloaders_dict['test'], datasets_dict['test'], model_name, ds_dir, class_reports_dir, device, display_reports)
                print('done')

            if gradCAM_analysis:
                print('gradCAMs explainability ...', end=' ')
                # gradCAMs
                proceed_gradCAMs(model, dataloaders_dict['test'], datasets_dict['test'], model_name, ds_dir, device, gradCAM_dir, display_reports)
                print('done')

            hists += hist
        
        # Save the histories
        save_hists(hists, f'{model_dir}/hists.csv')

def predict(model, dataloader, device=DEVICE):
    """
    Function which use a model to predict labels.
    Parameters:
        - 'model': a troch.Model object. The model to use to make predictions ;
        - 'dataloader': a torch.utils.data.Dataloader object. The dataloader 
                        supplying test images to the model ;
        - 'device': string, optional. The device (cuda or cpu) to use to 
                    train the model. Default to the device detected by the 
                    script ;
    Return: the list of the true labels and the list of the predicted ones.
    """

    preds = []
    y_test = []

    model = model.to(device)

    # Make predictions on test set
    for X, Y in dataloader:
        X_test = X.to(device)
        y_test.append(Y)
        with torch.set_grad_enabled(False):
            pred = model(X_test)
        if isinstance(pred, torch.Tensor): preds.append(pred)
        else: preds.append(pred.logits)

    y_test = torch.cat(y_test, dim=0)

    preds = torch.cat(preds, dim=0)
    _, y_pred = torch.max(preds, 1)

    # Convert lists of tensors to list of integers
    y_test = [int(y.cpu().numpy()) for y in y_test]
    y_pred = [int(y.cpu().numpy()) for y in y_pred]

    return y_test, y_pred

def plot_histories(model_names, ds_dirs, root_dir=DATA_DIR, save_dir=None, display_img=True):
    """
    Function plotting on the same figure all the histories corresponding to 
    the couples of 'model_names'/'ds_dirs'.
    Parameters:
        - 'model_names': a list. The list of the trained models names ;
        - 'ds_dirs': a list. The list of the datasets names to use for 
                     training ;
        - 'root_dir': a string or pathlib.Path object, optional. The root 
                      directory where to find the histories. Default to 
                      DATA_DIR ;
        - 'save_dir': a string or pathlib.Path object, optional. The directory 
                      in which to save the figure. If None, the figure won't 
                      be saved. Default to None ;
        - 'display_img': a boolean. Whether to display the figure or not. 
                         Default to True.
    """

    metrics = ['loss', 'accuracy']
    steps = ['train', 'val']

    steps_styles = {'train': '--', 'val': '-'}

    colors = list(mcolors.TABLEAU_COLORS.values())[:len(ds_dirs)]
    ds_colors = {ds: color for ds, color in zip(ds_dirs, colors)}

    custom_lines = [Line2D([0], [0], color=c) for c in colors] + [Line2D([0], [0], color=colors[0], linestyle=l) for l in list(steps_styles.values())]
    legends = ds_dirs + steps

    fig, axs = plt.subplots(nrows=len(model_names), ncols=len(metrics), figsize=(24, 36))

    for i, model_name in enumerate(model_names):
        hists = load_hists(f'{root_dir}/models/{model_name}/hists.csv')
        hists = hists_to_dict(hists, ds_dirs)

        for ds in ds_dirs:
            hist = hists[ds]

            for j, metric in enumerate(metrics):
                n_epochs = len(hist[metric]['train'])
                epochs = range(1, n_epochs+1)
                for step in steps:
                    h = hist[metric][step]
                    axs[i, j].plot(epochs, h, color= ds_colors[ds], linestyle=steps_styles[step], label=ds)
                axs[i, j].set_xlabel('Epochs')
                axs[i, j].set_ylabel(metric.capitalize())
                axs[i, j].legend(custom_lines, legends)
        axs[i, 0].set_title(model_name)

    if save_dir is None: pass
    else:
        fig.savefig(f'{save_dir}/pretrained_histories.png')

    if display_img: plt.show();

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main

if __name__ == "__main__":
    pass
