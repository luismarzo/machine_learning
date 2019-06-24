#Most machine learning libraries (including scikit-learn) give an error
#if you try to build a model using data with missing values. So you'll need
#to choose one of the strategies below.

    #1) A Simple Option: Drop Columns with Missing Values.The simplest option is 
    # to drop columns with missing values. (not good option)

    #2)Imputation fills in the missing values with some number. For instance,
    #we can fill in the mean value along each column.

    #3)mputation is the standard approach, and it usually works well.
    #  However, imputed values may be systematically above or below their actual 
    # values (which weren't collected in the dataset). Or rows with missing values
    #  may be unique in some other way. In that case, your model would make better 
    # predictions by considering which values were originally missing.



import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  # make the model
# validate the model (it says us the error)
from sklearn.metrics import mean_absolute_error
# split data into training and validation data, for both features and target
from sklearn.model_selection import train_test_split
# Using new model
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer



def main():
    # save filepath to variable for easier access
    file_path = '/home/luis/ws_machine_learning/machine_learning/intermediate/melb_data.csv'

    data_raw = pd.read_csv(file_path)

    # selecting the prediction target
    y = data_raw.Price

    # We do not include Price column
    data_predictors = data_raw.drop(['Price'], axis=1)
    # To keep things simple, we'll use only numerical predictors
    X = data_predictors.select_dtypes(exclude=['object'])

    # split the data into 2 groups for validate the model. We will validate it with diferent data
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, train_size=0.8, test_size=0.2)

    # ----------------------------Approach 1: Drop columns with missing values--------------------
    
    # Get names of columns with missing values
    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    # Drop columns in training and validation data
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_val.drop(cols_with_missing, axis=1)

    print("MAE from Approach 1 (Drop columns with missing values):")
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_val))

    # ---------------------------Approach 2: Imputation ----------------------------    

    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_val = pd.DataFrame(my_imputer.transform(X_val))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_val.columns = X_val.columns

    print("MAE from Approach 2 (Imputation):")
    print(score_dataset(imputed_X_train, imputed_X_val, y_train, y_val))

    # ---------------------------Approach 3: Extension to imputation----------------

    # Make copy to avoid changing original data (when imputing)
    X_train_plus = X_train.copy()
    X_val_plus = X_val.copy()

    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_val_plus[col + '_was_missing'] = X_val_plus[col].isnull()

    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_val_plus = pd.DataFrame(my_imputer.transform(X_val_plus))

    # Imputation removed column names; put them back
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_val_plus.columns = X_val_plus.columns

    print("MAE from Approach 3 (An Extension to Imputation):")
    print(score_dataset(imputed_X_train_plus, imputed_X_val_plus, y_train, y_val))

    # Shape of training data (num_rows, num_columns)
    print(X_train.shape)

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])



def score_dataset(X_train, X_val, y_train, y_val):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)


if __name__ == "__main__":
    main()
