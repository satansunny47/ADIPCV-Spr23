'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import DataLoader


import os
import argparse

from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class CustomCIFAR10(Dataset):
  def __init__(self, root, train=True, transform=None, download=True, train_split=0.8):
    super(CustomCIFAR10, self).__init__()
    self.dataset = CIFAR10(root, train=train, transform=transform, download=download)
    self.train_split = train_split

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    img, label = self.dataset[idx]
    return img, label

  def get_train_data(self):
    # Filter and return training data with 4800 images per class
    train_len = int(len(self) * self.train_split)
    class_counts = {}
    train_data = []
    for i in range(train_len):
      img, label = self[i]
      if label not in class_counts:
        class_counts[label] = 0
      if class_counts[label] < 4800:
        train_data.append((img, label))
        class_counts[label] += 1
    return train_data

  def get_test_data(self):
    # Filter and return testing data with 1200 images per class
    test_len = len(self) - int(len(self) * self.train_split)
    class_counts = {}
    test_data = []
    for i in range(int(len(self) * self.train_split), len(self)):
      img, label = self[i]
      if label not in class_counts:
        class_counts[label] = 0
      if class_counts[label] < 1200:
        test_data.append((img, label))
        class_counts[label] += 1
    return test_data
# #############################################################
  
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import Dataset, Subset
from collections import defaultdict

class CombinedCIFAR10(Dataset):
  def __init__(self, root, transform=None, download=True):
    super(CombinedCIFAR10, self).__init__()
    self.train_dataset = CIFAR10(root, train=True, transform=transform, download=download)
    self.test_dataset = CIFAR10(root, train=False, transform=transform, download=download)
    self.data = []
    self.combine_data()

  def combine_data(self):
    self.data.extend(self.train_dataset)
    self.data.extend(self.test_dataset)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img, label = self.data[idx]
    return img, label

def split_by_class(data, train_ratio=0.8):
  # Group data by class label
  class_data = defaultdict(list)
  for img, label in data:
    class_data[label].append((img, label))

  # Split data for each class
  train_data, test_data = [], []
  for label, class_items in class_data.items():
    split_idx = int(len(class_items) * train_ratio)
    train_data.extend(class_items[:split_idx])
    test_data.extend(class_items[split_idx:])

  return train_data, test_data

# Combine and split data
combined_dataset = CombinedCIFAR10(root='data/', transform=ToTensor(), download=True)
train_dataset, test_dataset = split_by_class(combined_dataset, train_ratio=0.8)

num_classes = 10  # Adjust this based on your dataset's number of classes
batch_size = 32

# Use train_data and test_data for training and testing
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = models.resnet50(pretrained=True)
# net = models.efficientnet_b0(pretrained=True)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
print('==> Finished Training')
print("Accuracy: ", best_acc)
