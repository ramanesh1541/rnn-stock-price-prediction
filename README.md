# <p align = "center"> Stock Price Prediction </p>

## AIM
To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
We are developing the RNN model to predict the stock prices of Google using the dataset provided. The dataset has many features, but we will be predicting the "Open" feauture alone. We will be using a sequence of 50 readings to predict the 51st reading.we have taken 50 Inputs with 50 Neurons in the RNN Layer (hidden) and one neuron for the Output Layer.These parameters can be changed as per requirements.

## Neural Network Model


![ex 5](https://user-images.githubusercontent.com/75235334/196046839-ff1bae2a-fde9-40d2-ae96-ef65b86db83a.png)

## DESIGN STEPS

### STEP 1:
Import tensorflow library and preprocessing libraries.

### STEP 2:
Load the traning dataset and take one column values and scale it using minmaxscaler.

### STEP 3:
Split x_train(0-50 values) and y_train(51 st value).

### STEP 4:
Create a RNN model with required no of neurons with one output neuron.

### STEP 5:
Fit the model and load testing dataset.For x_test,combine the values of both datasets.

### STEP 6:
Follow the same splitting.Make the prediction.

### STEP 7:
Plot graph and find the mse value.


## PROGRAM
```
# Developed By:RAMANESH R
# Register Number:212220040130
```
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('/content/trainset.csv')
dataset_train.columns
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
X_train_array = []
y_train_array = []
for i in range(50, 1259):
  X_train_array.append(training_set_scaled[i-50:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(50,1)))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('/content/testset.csv')
test_set = dataset_test.iloc[:,1:2].values
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test=[]
for i in range(50,1384):
  X_test.append(inputs_scaled[i-50:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(50,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)

```

## <br><br><br><br>OUTPUT

### True Stock Price, Predicted Stock Price vs time
![image](https://user-images.githubusercontent.com/75235334/195596872-3e362993-486d-400d-8ceb-913853aa2d10.png)

### Mean Square Error
![image](https://user-images.githubusercontent.com/75235334/195597795-27e9a7c9-b975-4560-ace6-54268a2d03db.png)

## RESULT
Thus the stock price prediction using RNN network is successfully implemented.
