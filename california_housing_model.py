import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import _california_housing as ch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator,	TransformerMixin
from sklearn.metrics import	mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


#Creating data set
dataset = ch.fetch_california_housing()
housing_data = pd.DataFrame(data= dataset.data, columns=dataset.feature_names)
housing_data["MedHouseValue"] = dataset.target
housing_data["RmBdrmRatio"] = housing_data["AveRooms"]/housing_data["AveBedrms"]
housing_data["PopBdrmRatio"] = housing_data["Population"]/housing_data["AveBedrms"]

"""
Columns in data:
MedInc
HouseAge
AveRooms
AveBedrms
Population
AveOccup
Latitude
Longitude
MedHouseValue
RmBdrmRatio (Feature engineered)
PopBdrmRatio (Feature engineered)
"""

class DataFrameSelector(BaseEstimator, TransformerMixin): #transformer utilized to convert Dataframe into useable sklearn 
	def	__init__(self, attribute_names):
		self.attribute_names = attribute_names
	def	fit(self, X, y=None):
		return	self
	def	transform(self, X):
		return	X[self.attribute_names].values

def split_sets(data, ratio): #to split data into training and testing set 
    shuffled_indices = np.random.permutation(len(data))
    test_set_size =	int(len(data) *	ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices =	shuffled_indices[test_set_size:]
    return	data.iloc[train_indices], data.iloc[test_indices]

training_set, testing_set = split_sets(housing_data, 0.2)
targets = training_set["MedHouseValue"]
new_targets = testing_set["MedHouseValue"]

corr_matrix = housing_data.corr()
"""
MedInc           0.688075 <-- Kept corr > 0.3
HouseAge         0.105623 <-- Kept due to small corr
AveRooms         0.151948 <-- Kept due to small corr
AveBedrms       -0.046701 <-- eliminated
Population      -0.024650 <-- eliminated
AveOccup        -0.023737 <-- eliminated
Latitude        -0.144160 <-- eliminated
Longitude       -0.045967 <-- eliminated
MedHouseValue    1.000000 <-- eliminated (since its the target)
RmBdrmRatio      0.383672 <-- Kept corr > 0.3
PopBdrmRatio    -0.016603 <-- eliminated
"""

num_attribs = ["MedInc", "RmBdrmRatio", "HouseAge", "AveRooms"] #Features with strong correlation
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),('std_scaler', StandardScaler())])#data pipeline used to transform housing
transformed_training_set = num_pipeline.fit_transform(training_set)
transformed_target_set = num_pipeline.fit_transform(testing_set)


#Custom Gradient Descent Model for Practice
def hypothesis(inputs, weights):#Actual function utilized linear regression since corr > 0.3 
    val = 0
    for i in range(len(inputs)):
        val += inputs[i] * weights[i]
    val += weights[-1]
    return val
        

def regression(targets, rate, all_inputs): #Homebrew gradient descent
    weights = [0] * (len(all_inputs[2]) + 1)
    for i in range(len(targets)):
        inputs = all_inputs[i]
        new_weights = []
        for k in range(len(inputs)):
            weight = weights[k] + rate * (targets.iloc[i] - hypothesis(inputs, weights)) * inputs[k]
            new_weights.append(weight)
        weight = weights[k + 1] + rate * (targets.iloc[i] - hypothesis(inputs, weights))
        new_weights.append(weight)
        weights = new_weights
    for i in range(len(targets)):
        inputs = all_inputs[i]
        new_weights = []
        for k in range(len(inputs)):
            weight = weights[k] + rate * (targets.iloc[i] - hypothesis(inputs, weights)) * inputs[k]
            new_weights.append(weight)
        weight = weights[k + 1] + rate * (targets.iloc[i] - hypothesis(inputs, weights))
        new_weights.append(weight)
        weights = new_weights
    return weights

def predict(all_inputs, weights):
    output = []
    for i in range(len(all_inputs)):
         inputs = all_inputs[i]
         output.append(hypothesis(inputs, weights))
    return output
         
      

#sklearn linear regression
lin_reg = LinearRegression()
lin_reg.fit(transformed_training_set, targets)
housing_predictions	=	lin_reg.predict(transformed_target_set)
lin_mse	= mean_squared_error(new_targets, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

#homebrew
weights = regression(targets, 0.001/len(transformed_training_set[0]), transformed_training_set)
home_housing_predictions = predict(transformed_target_set, weights)
new_lin_mse	= mean_squared_error(new_targets, home_housing_predictions)
new_lin_rmse = np.sqrt(new_lin_mse)

#sklearn descision tree regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(transformed_training_set, targets)
housing_predictions	= tree_reg.predict(transformed_target_set)
tree_mse = mean_squared_error(new_targets, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

#sklearn random forest regressor
forest_reg = RandomForestRegressor()
forest_reg.fit(transformed_training_set, targets)
housing_predictions = forest_reg.predict(transformed_target_set)
forest_mse = mean_squared_error(new_targets, housing_predictions)
forest_rmse = np.sqrt(forest_mse)


print(lin_rmse) #0.7735 <-- Baseline
print(new_lin_rmse) #0.9549 <-- Not bad could be better
print(tree_rmse)
print(forest_mse)

#Hyperparameter finetuning 
param_grid = [
     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

# grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring='neg_mean_squared_error')
# grid_search.fit(transformed_training_set, targets)
# print(grid_search.best_params_)
#n_estimators = 30, max_features = 2