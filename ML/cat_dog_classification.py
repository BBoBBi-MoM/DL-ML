#%% importing library
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

#%% hyper parameters

batch_size =32
learning_rate = 0.0001
traing_epoch =25

#%% dataset

transform = transforms.Compose([transforms.Resize((128,128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])

train_set = torchvision.datasets.ImageFolder(root=r'C:\Users\Administrator\Desktop\python_1\DL-ML\dataset\cat_dog\training_set',
                                             transform = transform)
test_set = torchvision.datasets.ImageFolder(root=r'C:\Users\Administrator\Desktop\python_1\DL-ML\dataset\cat_dog\test_set',
                                            transform = transform)

data_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,drop_last=True)

#%%
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)    
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)    
            )
        self.fc1 = nn.Linear(512,512,bias =True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512,2,bias =True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self,x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = output.view(output.size(0),-1)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

total_batch = len(data_loader)
print('Learnig start')
for epoch in range(traing_epoch):
    average_cost = 0
    for num,data in enumerate(data_loader):
        imgs ,labels = data
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()

        
        average_cost += loss / total_batch
        break
    print('[Epoch:{}] cost = {}'.format(epoch+1,average_cost))
    
print('finished')
# %%
