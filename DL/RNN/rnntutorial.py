#%% importing libraries
import torch
import numpy as np

#%% setting parameters
input_size = 4    #입력사이즈 단어 수 등등..
hidden_size = 2   #출력사이즈와 같다 
# %% one-hot-encoding
h=[1,0,0,0]
e=[0,1,0,0]
l=[0,0,1,0]
o=[0,0,0,1]
#%% 
input_data_np = np.array([[h,e,l,l,o],
                          [e,o,l,l,l],
                          [l,l,e,e,l]],dtype=np.float32)

#%% transform as torch tensor
input_data = torch.Tensor(input_data_np)
rnn = torch.nn.RNN(input_size,hidden_size)

outputs,_status = rnn(input_data)
#outputs : (batch_size,seq_size,hidden_size)
# %%
