import sys
import os
import re
from collections import OrderedDict
from os import listdir
from os.path import isfile, join
from importlib import reload
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import pandas
import torch
import matplotlib.pyplot as plt
import numpy as np
import Data
sys.path.append("../")
from Models.ResNet import ResnetClassifier
def return_analystics(model_path, training_analytics_path, num_classes, tag):
    training_analytics = torch.load(training_analytics_path)
    epochs = len(training_analytics[0])
    train_accuracy_array = training_analytics[0]
    top1_validation_acc_array = training_analytics[1]
    top3_validation_acc_array = training_analytics[2]
    
    
    # Load and evaluate the model
    print(f"Attempting to load model from path: {model_path}")
    model = ResnetClassifier(num_classes=num_classes)
    model.eval()
    dict = torch.load(model_path)
    model_state_dict = dict['model']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' if it exists
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    # Prepare dataset
    dataset = Data.DiffractionDataset(classes,1e-3, "../Data/" + tag + '/' + tag + '_Test.pt', categorical='Bravais Lattice')

    # Perform prediction and calculate accuracy
    supervised_output = model(dataset.data, labels=dataset.labels.long())
    top1_test_accuracy = supervised_output.accuracy(dataset.labels)
    top3_test_accuracy = supervised_output.top_k_acc(dataset.labels,3)
    top1_predictions = supervised_output.predictions
    top3_predictions = supervised_output.top_k_preds(3)

    return epochs, train_accuracy_array, top1_validation_acc_array, top3_validation_acc_array, top1_test_accuracy, top3_test_accuracy, top1_predictions, top3_predictions

    
