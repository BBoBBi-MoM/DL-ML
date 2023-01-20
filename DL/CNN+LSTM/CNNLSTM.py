#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
covid_19=pd.read_csv(r'./dataset/covid_19_data.csv')
ts_covid_19 = pd.read_csv(r'./dataset/time_series_covid19_confirmed_global.csv')
# %%
korea_data=ts_covid_19[ts_covid_19['Country/Region']=='Korea, South'].iloc[:,4:].T
korea_data.index = pd.to_datetime(korea_data.index)
# %%
korea_data = korea_data.diff().fillna(korea_data.iloc[0]).astype('int')

korea_data 

# %% 
def data_to_sequence(data,step_size=5):
    t = []
    label = []
    for i in range(len(data)-step_size):
        t_value = data.iloc[i:i+step_size]
        t.append(t_value)
        label_value = data.iloc[i+step_size]
        label.append(label_value)
    return np.array(t) ,np.array(label)


# %%
X,y=data_to_sequence(korea_data)
train_size = int(len(X)*0.8)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]
# %% scailing
Max = X_train.max()
Min = X_train.min()

def MinMaxScailing(array,min,max):
    return (array - min)/(max-min)
X_train = MinMaxScailing(X_train,Min,Max)
X_test = MinMaxScailing(X_test,Min,Max)
y_train = MinMaxScailing(y_train,Min,Max)
y_test = MinMaxScailing(y_test,Min,Max)


# %%
