#%% importing libraries
import numpy as np
import matplotlib.pyplot as plt
#%% generating dataset
num_data = 2400
t = np.linspace(0.0,100,num_data)
y = np.sin(t) + np.sin(2*t)
plt.plot(t,y)

# %% make a sequential dataset
seq_len = 10
seq_data=list()
seq_label = list()
for idx in range(len(t)-seq_len):
    seq_data.append(y[idx:idx+seq_len])
    seq_label.append(y[idx+seq_len])

seq_data = np.array(seq_data)
seq_label = np.array(seq_label)

#seq_data = np.swapaxes(seq_data,0,1)
seq_data = np.transpose(seq_data)
seq_data = np.expand_dims(seq_data, axis=2)

# %% define the RNN class
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, output_dim,hid_dim, batch_size):
        super(RNN,self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.batch_size = batch_size

        self.u = nn.Linear(self.input_dim, self.hid_dim, bias=False)
        self.v = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.w = nn.Linear(self.hid_dim, self.output_dim, bias=False)
        self.act_func = nn.Tanh()

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.zeros(batch_size,self.hid_dim)
    
    def forward(self, x):
        h = self.act_func(self.u(x)+self.v(self.hidden))
        return y,h

#%% setting parameters
import torch.optim as optim
model = RNN(1,1,50,2390)
loss_fn = nn.MSELoss()
optimizer =optim.SGD(model.parameters(),lr=0.005)
training_epoch = 500
#%% training step

for i in range(training_epoch):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    model.hidden = model.init_hidden()

    for t in seq_data:
        t = torch.Tensor(t).float()
        seq_label = torch.Tensor(seq_label).float()

        predition, hidden = model(t)
        model.hidden = hidden

    loss = loss_fn(predition.view(-1),seq_label.view(-1))
    loss.backward()
    optimizer.step()
    print(loss.item())
# %%
