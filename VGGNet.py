import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageFile

#ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from time import sleep
import json
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 2 to use

batch_size=128

data_transforms = {
    "TRAIN": transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally.
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),
    "VAL": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),
    "TEST": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


trainset = torchvision.datasets.ImageFolder('/home/hjpark/dataset/imagenet/train',
                                         transform=data_transforms["TRAIN"])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

valset = torchvision.datasets.ImageFolder('/home/hjpark/dataset/imagenet/val',
                                         transform=data_transforms["VAL"])
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print("Loaded {} images under {}".format(len(train_loader.dataset), "train"))
print("Loaded {} images under {}".format(len(val_loader.dataset), "validation"))


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        self.features = features  # convolution

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )  # FC layer

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # Convolution
        x = self.avgpool(x)  # avgpool
        x = torch.flatten(x, 1)
        x = self.classifier(x)  # FC layer
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #8 + 3 =11 == vgg11
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # 10 + 3 = vgg 13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], #13 + 3 = vgg 16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] # 16 +3 =vgg 19
}



# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(val_loader):
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

            _, y_pred = outputs.data.topk(k=5, dim=1)  # _, [B, n_classes] -> [B, maxk]
            y_pred = y_pred.t()

            target_reshaped = labels.view(1, -1).expand_as(y_pred)
            # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
            correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
            # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

            # -- get topk accuracy
            list_topk_correct = []  # idx is topk1, topk2, ... etc
            for k in (1,5):
                # get tensor of which topk answer was right
                ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
                # flatten it to help compute if we got it correct for each example in batch
                flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(
                    -1).float()  # [k, B] -> [kB]
                # get if we got it right for any of our top k prediction for each example in batch
                tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0,
                                                                                            keepdim=True)  # [kB] -> [1]
                # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
                # topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
                list_topk_correct.append(tot_correct_topk)
            # print("list_topk_correct (in current batch): ", list_topk_accs)

            return torch.Tensor(list_topk_correct)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):

    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_correct = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            #topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_correct.append(tot_correct_topk)
        #print("list_topk_correct (in current batch): ", list_topk_accs)
        
        return torch.Tensor(list_topk_correct)


best_accuracy = 0.0

device = torch.device('cuda')
print("The model will be running on", device, "device")

#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)



num_epochs = 100

save_path = "/home/hllee/ivpl-edu/classification/VGGNet/checkpoint/"
valid_early_stop = 0
valid_loss_min = np.Inf
EARLY_STOPPING_EPOCH = 10
since = time.time()

print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)
total_step_t = len(val_loader)


#initial_epoch = checkpoint['epoch']+1
#initial_epoch = 0
restart = True

if restart:  # for re-training
    checkpoint = torch.load(f'{save_path}saved-model-batch64.pth')
    initial_epoch = checkpoint['epoch'] + 1
    val_loss = checkpoint['val_loss']
    val_acc = checkpoint['val_acc']
    train_loss = checkpoint['train_loss']
    train_acc = checkpoint['train_acc']

    print(val_loss)
    print(val_acc)
    print(train_loss)
    print(train_acc)
    print("initial_epoch", initial_epoch)
    #model = VGG(make_layers(cfg['D']), num_classes=1000, init_weights=True)

    model = VGG(make_layers(cfg['D']), num_classes=1000, init_weights=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, patience=3, cooldown=5, threshold=1e-3)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

else:
    model = VGG(make_layers(cfg['D']), num_classes=1000, init_weights=True)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, patience=3, cooldown=5, threshold=1e-3)

    print(model)
    model.to(device)
    initial_epoch=0


### Evaluation code

model.eval()

batch_loss = 0
total_t = 0
correct_t = 0

with torch.no_grad():
    model.eval()

    total_t = len(val_loader.dataset)
    topk_correct = torch.Tensor([0, 0])


    for i, (inputs_t, labels_t) in enumerate(tqdm(val_loader)):
        inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)
        outputs_t = model(inputs_t)
        loss_t = criterion(outputs_t, labels_t)
        topk_correct += accuracy(output=outputs_t, target=labels_t, topk=(1, 5))
        # print("topk_correct: ", topk_correct)
        batch_loss += loss_t.item()
    print("topk_correct: ", topk_correct)
    topk_acc = (topk_correct / total_t * 100).tolist()
    #val_acc.append(topk_acc)  # topk_acc = [top1_acc, top5_acc]
    #val_loss.append(batch_loss / total_step_t)

print(f'Test acc: top-1= {topk_acc[0]:.4f}% / top-5= {topk_acc[1]:.4f}%\n')




'''######## Start training!
for epoch in range(initial_epoch, num_epochs):  # loop over the dataset multiple times
    print(f' ====================== epoch %d ======================' % (epoch+ 1))
    print("Current learning rate: ", optimizer.param_groups[0]['lr'])
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    correct = 0
    total = 0

    # train
    total = len(train_loader.dataset)
    print("total train size: ", total)
    model.train()
    for i, data in enumerate(tqdm(train_loader)):
        #sleep(0.01)
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)

        # compute the accuracy over batch images
        correct += torch.sum(pred == labels).item()
        #total += labels.size(0)

        # print statistics
        if i % 2000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i, total_step, loss.item()))

    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f'\ntrain loss: {train_loss[-1]:.4f}, train acc: {train_acc[-1]:.4f}%')

    epoch_time = time.time() - since
    since = time.time()

    print('')
    print(f'[Summary] Elapsed time : %.0f m %.0f s' % (epoch_time // 60, epoch_time % 60))

    # valid
    batch_loss = 0
    total_t = 0
    correct_t = 0

    with torch.no_grad():
        model.eval()
        
        total_t = len(val_loader.dataset)
        topk_correct = torch.Tensor([0, 0])
        
        for i, (inputs_t, labels_t) in enumerate(tqdm(val_loader)):
            inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)
            outputs_t = model(inputs_t)
            loss_t = criterion(outputs_t, labels_t)
            topk_correct += accuracy(output=outputs_t, target=labels_t, topk=(1,5))
            #print("topk_correct: ", topk_correct)          
            batch_loss += loss_t.item()
        print("topk_correct: ", topk_correct)
        topk_acc = (topk_correct / total_t * 100).tolist()
        val_acc.append(topk_acc) # topk_acc = [top1_acc, top5_acc]
        val_loss.append(batch_loss / total_step_t)

        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {val_loss[-1]:.4f}, validation acc: top-1= {topk_acc[0]:.4f}% / top-5= {topk_acc[1]:.4f}%\n')
        lr_scheduler.step(val_loss[-1])


    torch.save({'epoch': epoch,
                'train_loss' : train_loss,
                'train_acc' : train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, f'{save_path}saved-model-batch64.pth') # 사실은 128
    print('Saving current model')

    if network_learned:
        valid_early_stop = 0
        best_model = model


    else:
        # early stopping
        valid_early_stop += 1
        if valid_early_stop >= EARLY_STOPPING_EPOCH:
            print("EARLY STOPPING!!")
            break


print('Finished Training')'''





