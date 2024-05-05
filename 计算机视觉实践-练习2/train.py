import torch
import torch.nn as nn
import torch.optim as optim
from model import LeNet
from torchvision import datasets,transforms
import os

data_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=False)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)
print("datasets downloaded")
# train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=data_transform, download=False)
# train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# # 加载测试数据集
# test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=data_transform, download=False)
# test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

# 学习率，每隔10轮变为原来的0.1
#lr_scheduler = lr_schedule.StepLR(optimizer, step_size=10, gamma=0.1)

def train(DataLoader,model,loss_fn,optimizer):
    loss,acc,n = 0.0, 0.0, 0

    for batch,(inputs,labels) in enumerate(DataLoader):
        inputs,labels = inputs.to(device),labels.to(device)
        
        outputs = model(inputs)
        cur_loss = loss_fn(outputs,labels)

        #计算准确率
        _,pred = torch.max(outputs,axis=1)
        cur_acc = torch.sum(labels==pred) / outputs.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        acc += cur_acc.item()
        n = n+1
    
    aver_loss = loss / n
    aver_acc = acc / n

    print('train_loss '+ str(aver_loss))
    print('train_acc '+ str(aver_acc))


def val(DataLoader,model,loss_fn):
    model.eval()
    loss,acc,n=0.0, 0.0, 0

    with torch.no_grad():
        for batch,(inputs,labels) in enumerate(DataLoader):
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            cur_loss = loss_fn(outputs,labels)

            _,pred = torch.max(outputs,axis=1)
            cur_acc = torch.sum(labels==pred) / outputs.shape[0]

            loss += cur_loss.item()
            acc += cur_acc.item()
            n = n + 1

        return acc / n



epoch = 10
min_acc = 0

for t in range(epoch):
    print(f'epoch {t + 1}\n-------------')

    train(train_dataloader,model,loss_fn,optimizer)

    acc = val(test_dataloader,model,loss_fn)

    if acc > min_acc :
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        
        min_acc=acc
        print("save best model")
        torch.save(model.state_dict(),'save_model/best_model.pth')


print("Done")

