# matplotlib inline  
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from fashion import FashionMNIST
import time


# ======================= IMPORTING DATA =========================
train_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

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

batch_size = 100
test_batch_size = 100

# Create dataloaders for both the train_data and the valid_data
train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_data,
    batch_size=batch_size, shuffle=True)

# Create dataloader for test data
test_loader = torch.utils.data.DataLoader(
    FashionMNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


def print_shape(x):
    if len(x.shape) == 1:
        print("x.shape [%d]" % (x.shape[0]))
    elif len(x.shape) == 2:
        print("x.shape [%d,%d]" % (x.shape[0], x.shape[1]))
    elif len(x.shape) == 3:
        print("x.shape [%d,%d,%d]" % (x.shape[0], x.shape[1], x.shape[2]))
    elif len(x.shape) == 4:
        print("x.shape [%d,%d,%d,%d]" % (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    elif len(x.shape) == 5:
        print("x.shape [%d,%d,%d,%d,%d]" % (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
    else:
        print("too many dims...")


# ==== Display an image in the dataset ====
# plt.imshow(train_loader.dataset.train_data[1].numpy())
# plt.show()
# plt.show(train_loader.dataset.train_data[10].numpy())
# plt.show()



# ==================== MODEL CLASS DEFINITION ======================
# Model (1) already given:
# Fully-connected network of 2 hidden layers
#   - 1st linear transformation layer using sigmoid activation
#   - 2nd linear transformation layer using softmax activation
#       (Gives probabilities on the target classes)
class FcNetwork(nn.Module):
    def __init__(self):
        super(FcNetwork,self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = torch.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

# Model (2):
# Convolutional neural-network of 2 hidden layers
#   - convolution layer using sigmoid activation
#   - linear transformation layer using softmax activation
class OneConvOneFCSig(nn.Module):
    def __init__(self):
        super(OneConvOneFCSig, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1, padding=2)
        self.fc1 = nn.Linear(28*28 * 20, 10)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = x.view(-1, 28*28 * 20)
        x = F.log_softmax(self.fc1(x), dim=1)
        return F.log_softmax(x, dim=1)

# Model (3):
# Convolutional neural-network of 2 hidden layers
#   - convolution layer using relu activation
#   - linear transformation layer using softmax activation
class OneConvOneFCRelu(nn.Module):
    def __init__(self):
        super(OneConvOneFCRelu, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1, padding=2)
        self.fc1 = nn.Linear(28*28 * 20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 28*28 * 20)
        x = F.log_softmax(self.fc1(x), dim=1)
        return F.log_softmax(x, dim=1)

# Model (4):
# Convolutional neural network of 3 hidden layers:
#   - Convolution layer using relu activation
#   - Pooling (downsampling) layer
#   - Fully-connected layer (linear transformation) with softmax activation
class OneConvDownsample(nn.Module):
    def __init__(self):
        super(OneConvDownsample, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1, padding=2)
        self.fc1 = nn.Linear(14*14 * 20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 14*14 * 20)
        return F.log_softmax(self.fc1(x), dim=1)

# Model (5):
# Convolutional neural network of 6 hidden layers:
#   - Convolution layer #1 using relu activation
#   - Pooling (downsampling) layer #1
#   - Convolution layer #2 using relu activation
#   - Pooling (downsampling) layer #2
#   - Fully-connected layer #1 (linear transformation) with relu activation
#   - Fully-connected layer #2 (linear transformation) with softmax activation
class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7 * 7 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ======================= MODEL TRAINING =========================
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    return model


def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in valid_loader:
          # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
          output = model(data)
          # valid_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
          valid_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
          pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
          correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader.dataset)
    print("valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * float(correct) / float(len(valid_loader.dataset))))
    return float(correct) / float(len(valid_loader.dataset)), valid_loss
    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
          # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
          # data, target = Variable(data, volatile=True), Variable(target) #deprecated
          output = model(data)
          # test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
          test_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
          pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
          correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / float(len(test_loader.dataset))))

    results[model.__class__.__name__].append(test_loss)
    results[model.__class__.__name__].append(100. * float(correct) / float(len(test_loader.dataset)))
    

# ======================= COMPARING MODELS =========================
def experiment(model, epochs=1, lr=0.001):
    print("Number of epochs: %i" % epochs)
    print("Learning rate: %f" % lr)
    best_precision = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    precisions = []
    start_time = time.time()

    epochs = range(1, epochs + 1)

    for epoch in epochs:
        model = train(model, train_loader, optimizer)
        precision, model_loss = valid(model, valid_loader)
        precisions.append(precision)
        losses.append(model_loss)
    
        if precision > best_precision:
            best_precision = precision
            best_model = model

    # Store execution time in seconds
    end_time = time.time()
    model_training_time = end_time - start_time

    results[model.__class__.__name__].append(model_training_time)

    plt.figure("Precisions")
    plt.plot(epochs, precisions, label=model.__class__.__name__)
    plt.figure("Losses")
    plt.plot(epochs, losses, label=model.__class__.__name__)

    return best_model, best_precision

best_precision = 0
models = [FcNetwork(), OneConvOneFCSig()] #, OneConvOneFCRelu(), OneConvDownsample(), DeepNet()] # add your models in the list
results = {}
for model in models:
    print("\n======================= Model: %s =====================" % model.__class__.__name__)
    # model.cuda()  # if you have access to a gpu
    
    results[model.__class__.__name__] = []
    plt.figure("Precisions")
    plt.xlabel('Epoch number')
    plt.ylabel('Precision')

    plt.figure("Losses")
    plt.xlabel('Epoch number')
    plt.ylabel('Negative Log-likelihood')
    model, precision = experiment(model)
    test(model, test_loader)
    if precision > best_precision:
        best_precision = precision
        best_model = model


print("\n======================= Model Performances =======================")
print("|%-25s|%-12s|%-12s|%-12s|" % ("Model","Time","Loss","Accuracy"))
print("----------------------------------------------------------------")
for model, values in results.items():
  print("|%-20s|%*.3f|%*.4f|%*.2f" % (model, 12, values[0], 12, values[1],12, values[2]))



# save plot
plt.figure("Precisions")
plt.legend(loc='lower right')
plt.savefig("PrecisionsPlot.png")
plt.figure("Losses")
plt.legend(loc='upper right')
plt.savefig("LossPlot.png")

plt.show()



