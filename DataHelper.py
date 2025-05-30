import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
#Creates and retruns path to Test set, and rest of Dataset
#return train tensor, and  test tensor, option to Save files as saveName
def dictionaryToTensor(pathToTrain, pathToTest,pathToValid=None, save=False, saveName = None):
    train = torch.load(pathToTrain)
    
    if (pathToTest != None):
        test = torch.load(pathToTest) 
    else:
        test = torch.tensor([])
        
    if(pathToValid != None):
        valid  = torch.load(pathToValid)
    else:
        valid = torch.tensor([])
    
    train['X'].shape #torch.Size([124470, 1, 3041])
    test['X'].shape #torch.Size([13820, 1, 3041])
    if(pathToValid != None):
        DataFeatures = torch.cat((train['X'],test['X'],valid['X']),dim=0)
    else:
        DataFeatures = torch.cat((train['X'],test['X']),dim=0)

    if(pathToValid != None):
        DataLabels = torch.cat((train['Y'],test['Y'],valid['Y']),dim=0)
    else:
        DataLabels = torch.cat((train['Y'],test['Y']),dim=0)
    
    DataLabels.shape #torch.Size([152120])
    DataFeatures.shape #torch.Size([152120, 1, 3041])
    if(save):
        torch.save(DataFeatures,saveName+'Features.pt')
        torch.save(DataLabels,saveName+'Labels.pt')
    return ICSDFeatures,ICSDLabels



#return train_x, train_y, test_x,test_y, returns random_state as well
#option to take in paths, if paths=True, it will treat dataTensor and labelTensor as paths
def createTrainAndValidation(path, validationSize, savePath, exper):
    # Check if path is a string and file exists
    if not isinstance(path, str) or not os.path.isfile(path):
        raise ValueError(f"Invalid path: {path}")
    
    # Load tensors from file
    try:
        dataTensor = torch.load(path)['X']
        labelTensor = torch.load(path)['Y']
    except Exception as e:
        raise RuntimeError(f"Error loading tensors from {path}: {str(e)}")

    # Convert tensors to numpy arrays
    data_np = dataTensor.numpy()
    labels_np = labelTensor.numpy()
    
    # Split data into train and test sets
    data_train_np, data_test_np, labels_train_np, labels_test_np = train_test_split(
        data_np, labels_np, test_size=validationSize, random_state=42
    )
    
    # Convert numpy arrays back to tensors
    data_train_tensor = torch.from_numpy(data_train_np)
    data_test_tensor = torch.from_numpy(data_test_np)
    labels_train_tensor = torch.from_numpy(labels_train_np)
    labels_test_tensor = torch.from_numpy(labels_test_np)
    
    # Save tensors to files
    dTrainPath = os.path.join(savePath, exper + '_Train.pt')
    dValdPath = os.path.join(savePath, exper + '_Validation.pt')
    
    try:
        torch.save({'X': data_train_tensor, 'Y': labels_train_tensor}, dTrainPath)
        torch.save({'X': data_test_tensor, 'Y': labels_test_tensor}, dValdPath)
    except Exception as e:
        raise RuntimeError(f"Error saving tensors to {dTrainPath} or {dValdPath}: {str(e)}")

    # Check the shapes to ensure they match the original dimensions
    print("Train data shape:", data_train_tensor.shape)
    print("Test data shape:", data_test_tensor.shape)
    print("Train labels shape:", labels_train_tensor.shape)
    print("Test labels shape:", labels_test_tensor.shape)
    
    return [dTrainPath, dValdPath]


#ICSDFeatures,ICSDLabels = dictionaryToTensor(pathToTrain,pathToTest,pathToEval)
#createTestTrainTensors(ICSDFeatures,ICSDLabels,0.3, paths=False)
