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
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)
A2 = np.dot(Z1,W2)+B2
Z2 = sigmoid(A2)
# %%
def identify_function(x):
    return x
#%%
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2,W3)+B3
# %%
Y = identify_function(A3)
# %%
def init_networt():
    network = dict()
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network
def forward(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = identify_function(a3)
    return y
#%%
network = init_networt()
x = np.array([1.,0.5])
y = forward(network,x)
# %%
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    
    return y

# %%
def sum_squares_error(y,t):
    return 0.5* np.sum((y-t)**2)
#%%
def cross_entropy_error(y,t):
    eps = 1e-7
    return -np.sum(t*np.log(y+eps))

# %%
