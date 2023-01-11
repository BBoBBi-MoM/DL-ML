#%% 라이브러리 호출
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 배열 생성
X = np.arange(1,11)
Y = np.array([3,5,8,4,9,7,12,8,9,13])
#%% 산점도 확인
plt.scatter(X,Y)
plt.show()
#%% 파라미터
W = 0
b = 0
batch_size = len(X)     # 벡터 X의 원소 개수
epoch = 5000
learning_rate = 0.001
#%% 학습
for i in range(epoch):
    hyphothesis = W*X +b
    cost = (np.sum(hyphothesis-Y)**2)/len(X)

    gradient_W = np.sum(W*(X**2)+(b*X)-(Y*X))*(2/batch_size)
    gradient_b = np.sum((W*X)+b-Y)*(2/batch_size)

    W -= gradient_W *learning_rate
    b -= gradient_b *learning_rate

    if i% 10 == 0:
        print(f'W:{round(W,3)},  {round(b,3)}:b,   cost:{round(cost,3)},    EPOCH:{i}/{epoch}')

# %%

plt.xlim(-1,15)
plt.ylim(0,20)
plt.scatter(X,Y,color='red')

plt.scatter(11,11*W+b, color='green')
plt.plot(X, W*X+b)

plt.show()

# %%
