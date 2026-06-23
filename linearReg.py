import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import _california_housing as ch
from sklearn.linear_model import LinearRegression

model = LinearRegression()
dataset = ch.fetch_california_housing()
df = pd.DataFrame(data= dataset.data, columns=dataset.feature_names)
df["MedHouseValue"] = dataset.target
print(df.head())
input_feature = df["MedInc"]
model.fit(input_feature.to_frame("MedInc"), dataset.target)

def hypothesis(input, weight1, weight2):
    return  input * weight2 + weight1

def regression(targets, rate, inputs, weight1, weight2):
    w1 = weight1
    w2 = weight2 

    for i in range(len(targets)):
        new_w2 = w2 + rate * (targets[i] - hypothesis(inputs[i], w1, w2)) * inputs[i]
        new_w1 = w1 + rate * (targets[i] - hypothesis(inputs[i], w1, w2))

        w1 = new_w1
        w2 = new_w2 


        
    
    return w1, w2

w1, w2 = regression(dataset.target, 1/len(input_feature), input_feature, 0, 0)
x_line = np.linspace(np.min(input_feature), np.max(input_feature), 100)
y_line = w2 * x_line + w1

m = model.coef_[0]

x_line1 = np.linspace(np.min(input_feature), np.max(input_feature), 100)
y_line1 = m * x_line


plt.scatter(input_feature, dataset.target)
plt.plot(x_line, y_line, color='red')
plt.plot(x_line1, y_line1, color='blue')
plt.show()