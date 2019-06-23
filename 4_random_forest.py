#The random forest uses many trees, and it makes a prediction by averaging the predictions 
#of each component tree. It generally has much better predictive accuracy than a single 
#decision tree and it works well with default parameters

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  # make the model
# validate the model (it says us the error)
from sklearn.metrics import mean_absolute_error
# split data into training and validation data, for both features and target
from sklearn.model_selection import train_test_split
# Using new model
from sklearn.ensemble import RandomForestRegressor


def main():
    # save filepath to variable for easier access
    file_path = '/home/luis/ws_machine_learning/machine_learning/ejemplo.csv'

    data_raw = pd.read_csv(file_path)

    # Filter the list. It eliminates void rows
    data = data_raw.dropna(axis=0)

    # selecting the prediction target
    y = data.longitude

    # Chosing features
    features = ['district', 'latitude', 'ucr_ncic_code']
    X = data[features]
    # split the data into 2 groups for validate the model. We will validate it with diferent data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    # leafs of the tree
    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

    best_tree_size = get_best_leaf(
        candidate_max_leaf_nodes, train_X, val_X, train_y, val_y)
    #print(best_tree_size)
    
    # For the final model, now you are going to use ALL the data that you have so:
    model = DecisionTreeRegressor(
        max_leaf_nodes=best_tree_size, random_state=0)
    model.fit(train_X, train_y)
    val_predict = model.predict(val_X)
    decision_mae=mean_absolute_error(val_predict,val_y)
    print("Validation MAE for best value of DecisionTreeRegressor:", decision_mae)

    # Now we are going to do a RandomForest model
    model = RandomForestRegressor(random_state=0)
    model.fit(train_X, train_y)
    val_predict = model.predict(val_X)
    decision_mae=mean_absolute_error(val_predict,val_y)
    print("Validation MAE for best value of RandomTreeRegressor:", decision_mae)    


    


def get_best_leaf(candidate_max_leaf_nodes, train_X, val_X, train_y, val_y):
    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    lenght = len(candidate_max_leaf_nodes)
    cnt_mae = np.zeros(lenght)
    cnt = 0
    # Using all the leafs and getting in a vector the MAE
    for max_leaf_nodes in candidate_max_leaf_nodes:
        cnt_mae[cnt] = get_mae_leafs(max_leaf_nodes, train_X, val_X, train_y, val_y)
        cnt = cnt+1

    # Result of the leaf with least MAE
    for cnt_array in range(lenght):
        if min(cnt_mae) == cnt_mae[cnt_array]:
            best_tree_size = candidate_max_leaf_nodes[cnt_array]
    return(best_tree_size)


def get_mae_leafs(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


if __name__ == "__main__":
    main()