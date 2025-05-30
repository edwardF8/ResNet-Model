import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
 
def check_for_nans(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}")
        return True
    return False
 
class DiffractionDataset(Dataset):
    def __init__(self,classes, background, data, labels=None, unsupervised=False, categorical='Bravais Lattice'):
        print("Attempting to load data")
        assert categorical=='Bravais Lattice' or categorical=='Space Group', "The key word argument categorical should be Bravais Lattice or Space Group, not {}".format(categorical)
        data=torch.load(data)
        self.data = data['X'] 
        self.labels= data['Y']
        if self.data.ndim == 2:
            print("Data is missing channel dimension; adding it.")
            self.data = self.data.unsqueeze(1)  # [N, 3041] -> [N, 1, 3041]
        print("Final data shape in dataset:", self.data.shape)
        self.categorical=categorical
        self.data = self.data.add(background)
        '''if classes == 144:
            classes_tensor = torch.tensor([
                 0,   1,   2,   3,   4,   6,   7,   8,   9,  10,  11,  12,  13,  14,
                17,  18,  19,  22,  25,  28,  29,  30,  31,  32,  33,  35,  37,  42,
                43,  44,  45,  46,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
                75,  77,  78,  79,  80,  81,  83,  84,  85,  86,  87,  91,  93,  95,
                97,  98, 106, 109, 112, 113, 114, 117, 120, 121, 122, 125, 126, 127, 128,
               129, 130, 134, 135, 136, 137, 138, 139, 140, 141, 145, 146, 147, 149,
               154, 159, 160, 163, 164, 165, 166, 168, 169, 172, 173, 175, 177, 178,
               179, 181, 184, 185, 186, 188, 189, 190, 191, 192, 193, 196, 197, 199,
               202, 203, 204, 205, 211, 212, 214, 215, 216, 217, 219, 220, 222, 224,
               225, 226, 228, 229])
            unique_classes = classes_tensor.tolist()
            class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
            self.labels = torch.tensor([class_mapping[label] for label in self.labels.tolist()], dtype=torch.long)
            self.mapping = class_mapping 
        #else:
            #self.mapping=torch.load('mapping.pt')[categorical]'''

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def batch_u(self, batch_size):
        idx = torch.randint(0,len(self.data),(batch_size,))
        return self.data[idx]

    def plot(self, idx=0, norm=False):
        assert idx<len(self.data), "Index out of range, please input an index between 0 and {}".format(len(self))
        data=self.data[idx].cpu().numpy().flatten()
        if(norm):
            data = data-20
            data = data/data.max() + 0.001
        xrange=np.arange(3, data.shape[-1]*0.05+3, 0.05)
        fig = plt.plot(xrange,data)
        plt.xlabel("2\u03B8\xb0")
        plt.ylabel("Normalized Intensity")
        str_label=self.mapping[int(self.labels[idx].item())]
        plt.title("{}: {}".format(self.categorical,str_label))
        return fig

    def _unencode_labels(self, labels):
        return np.asarray([self.mapping[int(i.item())] for i in labels])

    def compare(self, pred1, pred2=None, heading=[]):
        pred1=self._unencode_labels(pred1)
        labels=self._unencode_labels(self.labels)

        if pred2 is not None:
            pred2=self._unencode_labels(pred2)
            print("{: >3}{: >20}{: >20}{: >20}".format("Index", "True Label", heading[0], heading[1]))
            for i in range(len(pred1)):
                print("{: >3} {: >20} {: >20} {: >20}".format(i, labels[i], pred1[i], pred2[i]))

        else:            
            print("{: >3}{: >20}{: >20}".format("Index", "True Label", "Prediction"))
            for i in range(len(pred1)):
                print("{: >3} {: >20} {: >20}".format(i, labels[i], pred1[i]))

 