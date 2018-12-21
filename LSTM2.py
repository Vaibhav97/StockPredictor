import matplotlib.pyplot as plt
import statsmodels.tsa.seasonal as smt
import numpy as np 
import pandas as pd 
import random
import datetime as dt
from sklearn import linear_model 
from sklearn.metrics import mean_absolute_error
import plotly

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

import os
from subprocess import check_output

filenames = ['output7.csv']
data = []
filename=filenames[0]

df = pd.read_csv(filename, sep=',')
df['Label'] = filename
df['Date'] = pd.to_datetime(df['Date'])
data.append(df)
#________________________________________________________________________________
r = lambda: random.randint(0,255)
traces = []
for df in data:
    clr = str(r()) + str(r()) + str(r())
#     df = df.sample(n=100, replace=True)
    df = df.sort_values('Date')
#     print(df['Label'])
    label = df['Label'].iloc[0]
    trace = plotly.graph_objs.Scattergl(
        x=df['Date'],
        y=df['Close'],
        mode='line',
        line=dict(
            color = clr
        )
    )
    traces.append(trace)
    
layout = plotly.graph_objs.Layout(
    title='Plot',
)
fig = plotly.graph_objs.Figure(data=traces, layout=layout)
plotly.offline.init_notebook_mode(connected=True)
plotly.offline.iplot(fig, filename='dataplot')
#________________________________________________________________________________
df = data[0]
window_len = 10

split_date = list(data[0]["Date"][-(2*window_len+1):])[0]

training_set, test_set = df[df['Date'] < split_date], df[df['Date'] >= split_date]
training_set = training_set.drop(['Date', 'Label'], 1)
test_set = test_set.drop(['Date', 'Label'], 1)
#Create windows for training
LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    
    for col in list(temp_set):
        temp_set[col] = temp_set[col]/temp_set[col].iloc[0] - 1
    
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)
#Create windows for testing
LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    
    for col in list(temp_set):
        temp_set[col] = temp_set[col]/temp_set[col].iloc[0] - 1
    
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1
LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)
#________________________________________________________________________________
def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.10, loss="mae", optimizer="adam"):
    
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model
#________________________________________________________________________________
nn_model = build_model(LSTM_training_inputs, output_size=1, neurons = 64)
nn_history = nn_model.fit(LSTM_training_inputs, LSTM_training_outputs,epochs=5, batch_size=1, verbose=2, shuffle=True)
#________________________________________________________________________________
plt.plot(LSTM_test_outputs, label = "actual")
plt.plot(nn_model.predict(LSTM_test_inputs), label = "predicted")
plt.legend()
plt.show()
MAE = mean_absolute_error(LSTM_test_outputs, nn_model.predict(LSTM_test_inputs))
print('The Mean Absolute Error is: {}'.format(MAE))