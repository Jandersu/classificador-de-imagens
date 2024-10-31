
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torchvision import models
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import timm

"""# DATASET"""

main_dir = ""

os.chdir(main_dir)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_dataset = datasets.ImageFolder(root=main_dir, transform=transform)

image_dataset.classes

class Data(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.instances = self.make_instances()

    def make_instances(self):
        instances = []
        for class_name in self.classes:
            if class_name not in self.class_to_idx:
                continue
            class_dir = os.path.join(self.dir, class_name)
            for root, _, fnames in os.walk(class_dir):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    item = path, self.class_to_idx[class_name]
                    instances.append(item)
        return instances

    def __getitem__(self, index):
        path, target = self.instances[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.instances)

"""# LOADER"""

def image_loader(img):
    return Image.open(img).convert("RGB")

#img = image_loader('./controle-xbox/20241027_100645.jpg')

tconvert = transforms.Resize((255,255))

tconvert(img)

tconvert2 =  transforms.CenterCrop((227,227))

img_tensor = transform(img)

"""# Treinamento model EfficientNet B0


"""

ds = image_dataset

ds.targets = np.array(ds.targets)

bs = 64
train_idx, temp_idx = train_test_split(np.arange(len(ds)),test_size=0.3,shuffle=True,stratify=ds.targets)
valid_idx, test_idx = train_test_split(temp_idx,test_size=0.5,shuffle=True,stratify=ds.targets[temp_idx])

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
test_sampler  = torch.utils.data.SubsetRandomSampler(test_idx)

dl_train = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=train_sampler)
dl_valid = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=valid_sampler)
dl_test  = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=test_sampler)

x,y = next(iter(dl_train))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True)

#model = timm.create_model('lcnet_050.ra2_in1k', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.to(device)

loss_train = []
loss_eval  = []

patience_time = 15
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(),lr=0.01)

epochs = 20

stop = False
epoch = 0
lowest_loss_eval = 10000
last_best_result = 0
while (not stop):
    #wandb.watch(model)
    model.train()
    lloss = []
    for x,y in dl_train:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        closs = criterion(pred,y)
        closs.backward()
        opt.step()
        opt.zero_grad()
        lloss.append(closs.item())
        #print(closs.item())
    loss_train.append(np.mean(lloss))
    lloss = []
    model.eval()
    lres = []
    ytrue = []
    with torch.no_grad():
        for data,y in dl_valid:
            data = data.to(device)

            pred = model(data)
            closs = criterion(pred.cpu(),y)
            lloss.append(closs.item())
            res  = pred.argmax(dim=1).cpu().tolist()
            lres += res
            ytrue += y
    avg_loss_eval = np.mean(lloss)
    loss_eval.append(avg_loss_eval)
    #wandb.log({"loss_eval": avg_loss_eval,"loss_train":loss_train[-1]})
    if avg_loss_eval < lowest_loss_eval:
        lowest_loss_eval = avg_loss_eval
        last_best_result = 0
        print("Best model found! saving...")
        actual_state = {'optim':opt.state_dict(),'model':model.state_dict(),'epoch':epoch,'loss_train':loss_train,'loss_eval':loss_eval}
        #torch.save(model.state_dict(), 'best_model.pth')
        torch.save(actual_state,'best_model.pth')
    last_best_result += 1
    if last_best_result > patience_time:
        stop = True
    print("epoch %d loss_train %4.3f loss_eval %4.3f last_best %d"%(epoch,loss_train[-1],loss_eval[-1],last_best_result))
    epoch += 1

recover = torch.load('best_model.pth')
#actual_state = {'optim':opt.state_dict(),'model':model.state_dict(),'epoch':epoch}

opt.load_state_dict(recover['optim'])
model.load_state_dict(recover['model'])
loss_train = recover['loss_train']
loss_eval = recover['loss_eval']
epoch = recover['epoch']

l = []
for layer in model.children():
    l.append(layer)

#model.head = nn.Linear(1024,1)

model.eval()
lres = []
ytrue = []
with torch.no_grad():
    for data,target in dl_test:
        data = data.to(device)
        pred = model(data)
        res  = pred.argmax(dim=1).cpu().tolist()
        lres += res
        ytrue += target

plt.plot(loss_train, label='Train Loss')
plt.plot(loss_eval, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import classification_report

# Gera o relatório de classificação
print(classification_report(ytrue, lres, target_names=["abobora-halloween", "controle-xbox"]))