# Imports
import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from PIL import Image
from torchvision import models
from torch import optim  # For optimizers like SGD, Adam, etc.
# import torch_optimizer as optimi
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
from torchsummary import summary
import numpy as np
import pandas as pd
import os
import skimage as io
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import sys
import time

#Models
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_uniform_(m.weight, mode='fan_out')
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )
    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.celu(self.bn1(out), alpha=0.075)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
model = Wide_ResNet(28, 10, 0.3, 10)
model.apply(conv_init)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
# print(model)

# Hyperparameters
num_classes = 10
learning_rate = 0.01
batch_size = 128
num_epochs = 200

# Load Data
transform_train = transforms.Compose(
    [  # Compose makes it possible to have many transforms
        # transforms.ToPILImage(),
        transforms.Resize((36, 36)),  # Resizes (32,32) to (36,36)
        transforms.RandomCrop(32, padding=4),  # Takes a random (32,32) crop
        transforms.ColorJitter(brightness=0.5),  # Change brightness of image
        transforms.RandomRotation(
           degrees=55
        ),  # Perhaps a random rotation from -45 to 45 degrees
        # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        transforms.RandomHorizontalFlip(
          #  p=0.5
        ),  # Flips the image horizontally with probability 0.5
        transforms.RandomVerticalFlip(
            p=0.05
        ),  # Flips image vertically with probability 0.05
        transforms.RandomGrayscale(p=0.2),  # Converts to grayscale with probability 0.2
        transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
        # transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Note: these values aren't optimal / (value -mean)/std
    ]
)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_TTA = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(
           degrees=45
    ),  # Perhaps a random rotation from -45 to 45 degrees
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomHorizontalFlip(
      #  p=0.5
    ),  # Flips the image horizontally with probability 0.5
    transforms.RandomVerticalFlip(
        p=0.05
    ),  # Flips image vertically with probability 0.05
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = datasets.CIFAR10(root="/content/drive/MyDrive/COLAB/Learn Pytorch/data", train=True, transform=transform_train, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2)
test_dataset = datasets.CIFAR10(root="/content/drive/MyDrive/COLAB/Learn Pytorch/data", train=False, transform=transform_test, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers = 2)

#Augmentation
def get_mask(rate):
    image_shape = 32
    box_size = int(np.sqrt(1. - rate) * image_shape)
    x = np.random.randint(0, image_shape - box_size + 1)
    mask = torch.ones((3, image_shape, image_shape)).to(device)
    mask[:, x:(x+box_size), x:(x+box_size)] = 0
    return mask
def cutmix(x, y):
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(x.size()[0]).cuda()
    mask = get_mask(lam)

    mixed_x = x.mul(mask) + x[index, :].mul(1. - mask)
    mixed_y = y * lam + y[index, :] * (1 - lam)
    return mixed_x, mixed_y
def cutout(x, y):
    lam = np.random.uniform(low=0.75, high=0.96, size=None)

    mask = get_mask(lam)

    mixed_x = x.mul(mask)
    mixed_y = y * lam
    return mixed_x, mixed_y
def mixup(x, y):
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(x.size()[0]).cuda()

    mixed_x = x * lam + x[index, :] * (1 - lam)
    mixed_y = y * lam + y[index, :] * (1 - lam)
    return mixed_x, mixed_y
def Augmentation(inputs, targets):
    targets = 1.0 * F.one_hot(targets, num_classes=num_classes).float()
    flag = np.random.randint(2)
    if flag == 0: inputs, targets = mixup(inputs, targets)
    elif flag == 1: inputs, targets = cutmix(inputs, targets)
    else : inputs, targets = cutout(inputs, targets)
    # if flag == 0: inputs, targets = mixup(inputs, targets)
    return inputs, targets

#Loss and Optimizer
class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.2):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = 1.0 * labels * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()
criterion = nn.CrossEntropyLoss()
criterion_train = SmoothCrossEntropy()
optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.95, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, verbose = True)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
#                                                 total_steps=50 * 391, epochs=50, steps_per_epoch=391, pct_start=0.3, anneal_strategy='cos', 
#                                                 cycle_momentum=True, base_momentum=0.8, max_momentum=0.95, 
#                                                 div_factor=25.0, final_div_factor=10000.0, three_phase=False, last_epoch=-1, 
#                                                 verbose=True)

# Load checkpoint
print('| Resuming from checkpoint...')
checkpoint = torch.load( '/content/drive/MyDrive/COLAB/Learn Pytorch/data/wide_resnet (1).pth')
model.load_state_dict(checkpoint['model'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Augmentation(inputs, targets)
        
        #forawrd 
        outputs = model(inputs)
        loss = criterion_train(outputs, targets)
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # _, truth = targets.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(truth).sum().item()
        # progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        #backward
        optimizer.zero_grad()
        loss.backward()
        #gradient descent and adam step
        optimizer.step()
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(train_dataset)//batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()
    print(f"\nLoss: {train_loss / (batch_idx + 1):.5f}  Accuracy: {(100.*correct / total):.5f}%")


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.4f%%'
                %(epoch, num_epochs, batch_idx + 1,
                    (len(test_dataset)//100), loss.item(), 100.*correct/total))
            sys.stdout.flush()
        print(f"\nLoss: {test_loss / (batch_idx + 1):.5f}  Accuracy: {(100.*correct / total):.5f}%")

    # Save checkpoint.
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.4f%%" %(epoch, loss.item(), acc))
        if acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, '/content/drive/MyDrive/COLAB/Learn Pytorch/data/wide_resnet.pth')
            best_acc = acc

for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)
    scheduler.step()
