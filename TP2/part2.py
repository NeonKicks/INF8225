import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from fashion import FashionMNIST
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import numpy as np

#%matplotlib inline  


# Fetch data
train_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

# Get second copy of data
valid_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))


# Randomly select indexes that will be used as training examples
train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)



# Remove non-train examples from 'train_data'
train_data.train_data = train_data.train_data[train_idx, :]
train_data.train_labels = train_data.train_labels[torch.from_numpy(train_idx).type(torch.LongTensor)]

# Create a mask for all non-train data
mask = np.ones(60000)
mask[train_idx] = 0


# Remove train examples from valid_data
valid_data.train_data = valid_data.train_data[torch.from_numpy(np.argwhere(mask)), :].squeeze()
valid_data.train_labels = valid_data.train_labels[torch.from_numpy(mask).type(torch.ByteTensor)]


# Initialize batch size and test bacth size
batch_size = 100
test_batch_size = 100


# Create dataloaders for both the train_data and the valid_data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)


# Create dataloader for test data
test_loader = torch.utils.data.DataLoader(
    FashionMNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


# Show image
plt.imshow(train_loader.dataset.train_data[1].numpy())
plt.show()

#show second image
plt.imshow(train_loader.dataset.train_data[10].numpy())
plt.show()



