import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LiR
from prettytable import PrettyTable


def load_array(path, type):
    return np.load(f'{path}/pric_{type}.npy'), np.load(f'{path}/volu_{type}.npy')


def RMSE(pred, target):
    return np.sqrt(MSE(pred, target))


# print result
table = PrettyTable()
table.field_names = ['Model', 'Result']

# load array
path = './np_array/train_10_01_test_10_01'

train_x = np.concatenate((load_array(path, 'train_x')), axis=-1)
print(f"train_x.shape: {train_x.shape}")
train_y, _ = load_array(path, 'train_y')
print(f"train_y.shape: {train_y.shape}")
test_x = np.concatenate((load_array(path, 'test_x')), axis=-1)
print(f"test_x.shape: {test_x.shape}")
test_y, _ = load_array(path, 'test_y')
print(f"test_y.shape: {test_y.shape}")


# random forest
rf_model = RandomForestRegressor(max_depth=8, random_state=0, n_estimators=100)
rf_model.fit(train_x, train_y)
rf_result = RMSE(rf_model.predict(test_x), test_y)
table.add_row(["Random Forest", rf_result])
print(table)

# gradient boost
gb_model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=120, max_depth=9, min_samples_split=1200,
                                     min_samples_leaf=60, subsample=0.85, random_state=10, max_features='sqrt')
gb_model.fit(train_x, train_y)
gb_result = RMSE(gb_model.predict(test_x), test_y)
table.add_row(["Gradient Boost", gb_result])
print(table)

# linear regression
lir_model = LiR(normalize=True)
lir_model.fit(train_x, train_y)
lir_result = RMSE(lir_model.predict(test_x), test_y)
table.add_row(["Linear Regression", lir_result])
print(table)
