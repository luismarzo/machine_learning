import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# save filepath to variable for easier access
file_path = '/home/luis/ws_machine_learning/machine_learning/ejemplo.csv'

data = pd.read_csv(file_path)

#describe the data, the number of elements, desviacion tipica, etc
print(data.describe())

#It gives you the firts rows of the data like an example
print(data.head())

print(data.columns)

#selecting the predition target
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
print(model.predict(X))






