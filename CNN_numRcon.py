import torch
import torchvision
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=0.5, std=[0.5])])

path = "./dataset"

trainData = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
testData = torchvision.datasets.MNIST(path, train=False, transform=transform)

batch_size = 256
trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size)

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv = torch.nn.Sequential(
            # input size is 28 * 28
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 14 * 14 * 16
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 7 * 7 * 32
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # 7 * 7 * 64
            torch.nn.Flatten(),
            # 7 * 7 * 64(one dim)
            torch.nn.Linear(in_features=7*7*64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

model = MyModule()
loss_fc = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 10

def train(model, trainDataLoader, loss_fc, optimizer):
    model.train()
    loss_sum = 0
    cnt = 0
    for step, (trainImags, labels) in enumerate(trainDataLoader):
        trainImags = trainImags.to(device)
        labels = labels.to(device)
        y_ = model(trainImags)
        l = loss_fc(y_, labels)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss_sum += l.item()
        # print(l.item())
        cnt += 1
    return loss_sum / cnt

def test(model, testDataLoader, loss_fc):
    model.eval()
    loss_sum = 0
    cnt = 0
    accuracy = 0
    with torch.no_grad():
        for testImages, labels in testDataLoader:
            testImages = testImages.to(device)
            labels = labels.to(device)
            y_ = model(testImages)
            pred = y_.data.max(1, keepdim=True)[1]
            accuracy += pred.eq(labels.data.view_as(pred)).sum()
            l = loss_fc(y_, labels)
            loss_sum += l.item()
            cnt += 1
    return loss_sum / cnt, 100. * accuracy / len(testDataLoader.dataset)

loss_list = []
loss_list_test = []

for epoch in range(epochs):
    l1 = train(model, trainDataLoader, loss_fc, optimizer)
    l2, acc = test(model, testDataLoader, loss_fc)
    print("epoch:{}, train loss:{:.4f}, test loss:{:.4f}, acc:{:.4f}".format(epoch + 1, l1, l2, acc))
    loss_list.append(l1)
    loss_list_test.append(l2)

torch.save(model.state_dict(), "./module/CNN_numRcon.pth")

# 数据可视化
plt.plot(loss_list, label="train loss")
plt.plot(loss_list_test, label="test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

'''
epoch:1, train loss:2.2905, test loss:2.2704, acc:30.1300
epoch:2, train loss:2.1964, test loss:1.9776, acc:68.6900
epoch:3, train loss:1.0386, test loss:0.4883, acc:85.3800
epoch:4, train loss:0.4236, test loss:0.3509, acc:89.1500
epoch:5, train loss:0.3306, test loss:0.2773, acc:91.6100
epoch:6, train loss:0.2762, test loss:0.2525, acc:92.2900
epoch:7, train loss:0.2379, test loss:0.2288, acc:93.1600
epoch:8, train loss:0.2068, test loss:0.1892, acc:94.3500
epoch:9, train loss:0.1829, test loss:0.1579, acc:95.2000
epoch:10, train loss:0.1635, test loss:0.1502, acc:95.1400
'''
