#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
def AND(x1,x2):
    w1,w2,theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp<theta:
        return 0
    elif tmp > theta:
        return 1
#%%
AND(1,1)
# %%
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp <=0:
        return 0
    else:
        return 1
#%%
AND(0,1)
# %%
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x)+b
    if tmp <=0:
        return 0
    else:
        return 1
#%%
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <=0:
        return 0
    else:
        return 1
#%%
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y
#%%
# def step_function(x):
#     if x>0:
#         return 1
#     else:
#         return 0
# %% 배열입력
def step_function(x):
    y = x>0
    return y.astype(int)
#%%
x = np.linspace(-5,5,100)
y = step_function(x)
plt.plot(x,y)
plt.show()
# %%
def sigmoid(x):
    return 1/(1+np.exp(-x))
# %%
x = np.linspace(-10,10,50)
y = sigmoid(x)
plt.plot(x,y)
plt.show()
# %%
def relu(x):
    return np.maximum(0,x)
#%%
x = np.linspace(-5,5,100)
y = relu(x)
plt.plot(x,y)
plt.show()
# %%
X = np.array([1.,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X,W1)+B1
# %%
Z1 = sigmoid(A1)
# %%
Z1
# %%
