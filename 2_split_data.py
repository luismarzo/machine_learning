import pandas as pd
from sklearn.tree import DecisionTreeRegressor    #make the model
from sklearn.metrics import mean_absolute_error   #validate the model (it says us the error)
from sklearn.model_selection import train_test_split   # split data into training and validation data, for both features and target


# save filepath to variable for easier access
file_path = '/home/luis/ws_machine_learning/machine_learning/ejemplo.csv'

data_raw = pd.read_csv(file_path)

#Filter the list. It eliminates void rows
data = data_raw.dropna(axis=0)

#selecting the prediction target
y=data.longitude

#Chosing features
features = ['district' , 'latitude' , 'ucr_ncic_code']
X = data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
model = DecisionTreeRegressor()
model.fit(train_X , train_y)
prediction = model.predict(val_X)
MAE = mean_absolute_error(val_y, prediction)
print(MAE)

