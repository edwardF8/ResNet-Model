import os
import numpy as np
from datetime import datetime
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
import math
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from Models.ResNet import ResnetClassifier, ResnetConfig
from Data import DiffractionDataset
import DataHelper

class Logger:
    def __init__(self, tag, exper, time, epochs, batch_size, lr):
        self.time = time
        self.path = f"Testing Results/{exper}/Logger{self.time} | Total Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}.txt"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            f.write(f"Logger initialized at {self.time}\n")
            f.write(f"Total Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}\n\n")          
    def log(self, msg):
        with open(self.path, 'a') as f:
            f.write(msg + '\n')      
    def get_path(self):
        return self.path
    def getTime(self):
        return self.time
    
def init_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '7017'
    dist.init_process_group(backend, rank=rank, world_size=size)
    
def init_models(gpu, num_classes):
    config = ResnetConfig(
        input_dim = 1,
        output_dim = num_classes,
        res_dims=[32, 64, 64, 64],
        res_kernel=[5, 7, 17, 13],
        res_stride=[4, 4, 5, 3],
        num_blocks=[2, 2, 2, 2],
        first_kernel_size = 13,
        first_stride = 1,
        first_pool_kernel_size = 7,
        first_pool_stride = 7,
    )
    learning_rate = 0.5e-3
    model = ResnetClassifier(config)
    opt_SD=torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    return model, opt_SD, learning_rate

def train(gpu, epochs, world_size, batch_size, background, tensors, exper, tag, num_classes):
    # Initialize model
    setup_start = datetime.now()
    rank=gpu
    init_process(gpu, world_size)
    print(f"Rank {gpu + 1}/{world_size} process initialized.\n")
    torch.manual_seed(0)
    model, optimizer, lr = init_models(gpu, num_classes)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    torch.autograd.set_detect_anomaly(True)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True,static_graph=True)    
    
    #Loading Data
    cat = 'Bravais Lattice'
    if gpu == 0:
        test_dataset = DiffractionDataset(num_classes, background, tensors[1], unsupervised=False, categorical=cat)        
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
    supervised_dataset = DiffractionDataset(num_classes, background, tensors[0], unsupervised=False, categorical=cat)
    train_sampler=torch.utils.data.distributed.DistributedSampler(supervised_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=supervised_dataset, batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True,sampler=train_sampler)
    
   #Establish Training
    loss_function=torch.nn.CrossEntropyLoss().cuda(gpu)
    start = datetime.now()
    trainacc_array = []
    if gpu==0:
        print("Setup time: " + str(datetime.now() - setup_start))
        print("Model parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        L = Logger(tag, exper, str(start), epochs, batch_size, lr)
        time = L.getTime()
        model_path = f'Testing Results/{exper}/{tag}_Models_{time} | Total Epochs:{epochs}|Batch Size:{batch_size}|Learning Rate:{lr}.pt' 
        print(model_path)
        t1_array = []
        t3_array = []
    # Train cycle
    for epoch in range(epochs):
        t1_train_acc = 0
        train_examples = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.cuda(gpu, non_blocking=True) #makes sure stuff is on same device
            labels = labels.cuda(gpu, non_blocking=True)#makes sure stuff is on same device
            model.train() #sets model into training mode
            # Check for NaNs in imgs and labels
            if torch.isnan(imgs).any() or torch.isnan(labels).any():
                if gpu == 0:
                    print(f"[Epoch {epoch+1}/{epochs}] Skipping batch {i} due to NaN values in data or labels.")
                continue  # Skip this batch
            #Model Train + Update:  
            optimizer.zero_grad()
            train_output = model(imgs, labels=labels, s=True, loss_func=loss_function)
            supervised_accuracy = train_output.accuracy(labels)
            supervised_loss = train_output.loss
            supervised_loss.backward() 
            optimizer.step()
            #Logging:
            t1_train_acc += supervised_accuracy * imgs.size(0)
            train_examples += imgs.size(0)
            if gpu == 0:
                print("[Epoch %d/%d] [%d] [SD loss: %.2f  acc: %d%%]"
                      % (epoch+1, epochs, i, supervised_loss.item(), supervised_accuracy))
        t1_train_acc /= train_examples
        trainacc_array.append(t1_train_acc)
        #Evaluate and Update        
        if gpu==0:
            model.eval()
            top1_test_acc = 0
            top3_test_acc = 0
            total_loss = 0
            total_samples = 0
            for i, (test_data, test_labels) in enumerate(test_loader):    
                #Setup Data
                test_data=test_data.cuda(0)
                test_labels = test_labels.long().cuda(0)
                torch.cuda.empty_cache()
                
                test_output =model(test_data, labels=test_labels, s=True, loss_func=loss_function)
                batch_size = test_data.size(0)
                batch_t1_acc = test_output.accuracy(test_labels)
                batch_t3_acc = test_output.top_k_acc(test_labels,3)
                batch_loss = test_output.loss.item()
                
                #Computing weighted average 
                top1_test_acc += batch_t1_acc * batch_size
                top3_test_acc += batch_t3_acc * batch_size
                total_loss += batch_loss *batch_size
                total_samples += batch_size
                
            top1_test_acc /= total_samples
            top3_test_acc /= total_samples
            total_loss /= total_samples
            t1_array.append(top1_test_acc)
            t3_array.append(top3_test_acc)
            L.log("[Epoch %d/%d] [Train Acc: %d%%] [T1 Test Acc: %d%%] [Top 3 acc: %d%%]\n"
                  % (epoch+1, epochs, (t1_train_acc), (top1_test_acc), (top3_test_acc)))
            torch.cuda.empty_cache()
            torch.save({'model': model.state_dict(),'optimizer': optimizer.state_dict()}, model_path)
    if gpu == 0:
        torch.save(torch.tensor(np.stack([trainacc_array, t1_array, t3_array],axis=0)), f"Testing Results/{exper}/{tag}_train-t1_test-t3-test.pt")
def spawn(gpu, world_size, epochs, batch_size, background, tensors, exper, tag, num_classes, train=train):
    #args ={gpu, epochs, world_size, batch_size, background, tensors, exper, tag, num_classes}
    mp.spawn(
            train, args=(epochs, world_size, batch_size, background, tensors, exper, tag, num_classes),
            nprocs=world_size, join=True
        )

def createDir(tag, exper):
    td = f'Testing Results/{exper}/'
    dd = f'../Data/{exper}/' 
    os.makedirs(td,exist_ok=True)
    os.makedirs(dd,exist_ok=True)

if __name__=="__main__":
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL' # For detailed error messages. Feel free to remove
    tags = ['1 Epoch', '5 Epochs', '10 Epochs']
    erperiements = ['SanityCheck']
    epochs = [1]
    for i in range(1):
        exper = erperiements[i]
        tag = tags[i]
        createDir(tag,exper)
        epoch = epochs[i]
        classes = 230
        dataPath = '../Data/' + tag + '/' + tag + '_WoTest.pt' 
        savePath = '../Data/' + tag + '/'
        tensors =  DataHelper.createTrainAndValidation(dataPath,0.05,savePath,tag)
        torch.cuda.empty_cache()
        #args = gpu, world_size, epochs, batch_size, background, tensors, exper, tag, num_classes):
        spawn(4,4,epochs,500,1e-3,tensors,exper,tag,classes) #CHANGE THIS!!!!!!
        torch.cuda.empty_cache()
       # analytics(epoch,tag,exper,classes)
