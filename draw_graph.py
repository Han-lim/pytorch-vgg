import matplotlib.pyplot as plt
import torch

checkpoint = torch.load(f'/home/hllee/ivpl-edu/classification/VGGNet/checkpoint/saved-model-batch64.pth')
initial_epoch = checkpoint['epoch'] + 1
val_loss = checkpoint['val_loss']
val_acc = checkpoint['val_acc']
train_loss = checkpoint['train_loss']
train_acc = checkpoint['train_acc']

#print(val_loss)
#print(val_acc[:][0])
top1_val = []
top5_val = []

print(train_acc)

#print(train_acc[0])
#print(val_acc[1])
#print(train_acc[1])

#print("initial_epoch", initial_epoch)
# model = VGG(make_layers(cfg['D']), num_classes=1000, init_weights=True)

'''model = VGG(make_layers(cfg['D']), num_classes=1000, init_weights=True)
model.to(device)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, cooldown=5,
                                                    threshold=1e-3)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])'''

plt.figure(figsize=(10,5))
plt.title("Top1 Train Accuracy")
plt.plot(train_acc,label="train_acc")
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig('./Top-1 Train Accuracy')
plt.show()

