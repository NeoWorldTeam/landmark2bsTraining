import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import TensorDataset, Dataset, DataLoader



class FaceLandmarksDataset(Dataset):
    """面部标注数据集"""
    def __init__(self, filename, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.sample_size = input_size + output_size
        with open(filename, "rb") as f:
            self.landmarks_frame = np.fromfile(f, dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame) / (self.input_size + self.output_size)

    def __getitem__(self, index):

        newIndex = self.sample_size * index

        input_sample = self.landmarks_frame[newIndex:self.input_size]
        output_sample = self.landmarks_frame[(newIndex+self.input_size):self.output_size]

        input_tensor = torch.tensor(input_sample)
        output_tensor = torch.tensor(output_sample)

        return input_tensor,output_tensor
    

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        # x = torch.tanh(self.fc2(x))
        x = self.fc2(x)
        return x







# 实例化模型
input_size = 486*2
hidden_size = 256
output_size = 52
model = Model(input_size, hidden_size, output_size)


# 生成模拟数据
x_trains = torch.rand(50000, input_size) # 生成 100 组，每组有 486*2 个 float 变量
y_trains = torch.rand(50000, output_size) # 生成 100 组，每组有 52 个 0-1 范围的 float



# 把输入数据和标签数据打包成一个batch
train_data = TensorDataset(x_trains, y_trains)
# train_data = FaceLandmarksDataset("")
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters())

# define your Huber loss function
# criterion = nn.SmoothL1Loss()

# 开始训练
for i, data in enumerate(train_loader):
    inputs, labels = data
    # 梯度清零
    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pt")




# 加载模型
model = Model(input_size, hidden_size, output_size) # MyModel是你的模型类
model.load_state_dict(torch.load("model.pt"))
model.eval()

# 预测
with torch.no_grad():
    test_input = torch.rand(1, input_size)
    print(test_input)
    predicted_output = model(test_input)
    print(predicted_output)

    # predicted_output = torch.softmax(predicted_output,1)
    # _,predicted_label = torch.max(predicted_output,1)
    # print(predicted_label)


# #预测 
# # 准备好的预测数据
# x_test = ...

# # 预测
# with torch.no_grad():
#     outputs = model(x_test)
#     _, predicted = torch.max(outputs.data, 1)

# # outputs 是一个大小为 (batch_size, 52) 的张量，表示对于每个样本的 张量 个类别的预测概率。predicted 是一个大小为 (batch_size) 的张量，表示预测的类别