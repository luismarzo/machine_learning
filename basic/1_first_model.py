import pandas as pd
from sklearn.tree import DecisionTreeRegressor    #make the model
from sklearn.metrics import mean_absolute_error   #validate the model (it says us the error)

# save filepath to variable for easier access
file_path = '/home/luis/ws_machine_learning/machine_learning/basic/ejemplo.csv'

data_raw = pd.read_csv(file_path)

#Filter the list. It eliminates void rows
data = data_raw.dropna(axis=0)

#describe the data, the number of elements, desviacion tipica, etc
print(data.describe())

#It gives you the firts rows of the data like an example
print(data.head())

print(data.columns)

#selecting the prediction target
y=data.longitude

#Chosing features
features = ['district' , 'latitude' , 'ucr_ncic_code']
X = data[features]

#Building the model with sklearn. Steps:

	#Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
	#Fit: Capture patterns from provided data. This is the heart of modeling.
	#Predict: Just what it sounds like
	#Evaluate: Determine how accurate the model's predictions are.
	
#Making the model
model = DecisionTreeRegressor(random_state=1)
#Fit model
model.fit(X,y)
#Prediction
print('-----------prediction----------')
prediction = model.predict(X)
print(prediction)

#See the error
MAE = mean_absolute_error(y,prediction)
print(MAE)

#But the problem is that you are validating your model with the same data that you used before to build it, so this is wrong.
#We have to use a validation data, we have to save some data, build the model, and later validate it with this data that we dint use to build the model.







