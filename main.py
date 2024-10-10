import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

housing = pd.read_csv('housing.csv')
print(housing.info())
print(housing.describe())
print(housing.head())
print(housing['ocean_proximity'].value_counts())

#Filling missing data
housing['total_bedrooms'] = housing['total_bedrooms'].fillna(housing.groupby('ocean_proximity')['total_bedrooms']
                                                             .transform('median'))
#Transforming left skewed data
housing[['total_rooms','total_bedrooms','population','households','median_income','median_house_value']] = np.log1p(housing[['total_rooms','total_bedrooms','population','households','median_income','median_house_value']])

#Converting categorical data to numerical
columns = ['ocean_proximity']
final_data = pd.get_dummies(housing, columns=columns, drop_first=True)


#Splitting data
x_train,x_test,y_train,y_test = train_test_split(final_data.drop('median_house_value', axis=1),final_data['median_house_value'], test_size=.3, random_state=101)

#Scaling data
scaler = StandardScaler()
print(scaler.fit(X=x_train.drop(['longitude','latitude'], axis=1)))
x_train = scaler.transform(x_train.drop(['longitude','latitude',],axis=1))
x_test = scaler.transform(x_test.drop(['longitude','latitude'],axis=1))

#Hyperparameters for Decision Tree
dtree_params = {'max_depth': [64, 80,85,90,95,100], 'min_samples_split': [50], 'min_samples_leaf':[72,77,80,85,90,95,100]}

#Training the model
linear_regression = LinearRegression()
print(linear_regression.fit(x_train,y_train))
predictions = linear_regression.predict(x_test)

#Plotting residuals to check if they are normally distributed
sns.displot((y_test-predictions),bins=50, kde=True)
plt.show()

#Calculating performance metrics
print('Linear Regression Metrics')
print(metrics.root_mean_squared_error(y_test, predictions))
print(metrics.mean_absolute_error(y_test,predictions))
print(metrics.mean_squared_error(y_test,predictions))
print('R^2 score', metrics.r2_score(y_test,predictions))



#Decision Tree Regressor
d_tree = DecisionTreeRegressor()
d_tree.fit(x_train,y_train)
d_tree_predictions = d_tree.predict(x_test)
print('Parameters',d_tree.get_params())

#Calculating performance metrics
print('Decision Tree Regressor Metrics')
print(metrics.root_mean_squared_error(y_test, d_tree_predictions))
print(metrics.mean_absolute_error(y_test,d_tree_predictions))
print(metrics.mean_squared_error(y_test,d_tree_predictions))

#Plotting residuals to check if they are normally distributed
sns.displot((y_test-d_tree_predictions),bins=50, kde=True)
plt.show()

#Decision tree after hyperparameter tuning
grid = GridSearchCV(DecisionTreeRegressor(), dtree_params, refit=True, verbose=3)
grid.fit(x_train,y_train)

#Printing best params out of selected ones
print(grid.best_params_)
grid_predictions = grid.predict(x_test)
print('Predictions for model after hyperparameter tuning')
print('RMSE: ',metrics.root_mean_squared_error(y_test, grid_predictions))
print('MAE: ',metrics.mean_absolute_error(y_test,grid_predictions))
print('MSE: ',metrics.mean_squared_error(y_test,grid_predictions))
sns.displot((y_test-grid_predictions),bins=50, kde=True)
plt.title('Residuals of tuned D_Tree model')
plt.show()

#Random Forest Regressor
r_forest = RandomForestRegressor()
r_forest.fit(x_train,y_train)
r_forest_predictions = r_forest.predict(x_test)
print('Parameters',r_forest.get_params())
#Calculating performance metrics
print('Random Forest Regressor Metrics')
print('RMSE: ',metrics.root_mean_squared_error(y_test, r_forest_predictions))
print('MAE: ',metrics.mean_absolute_error(y_test,r_forest_predictions))
print('MSE: ',metrics.mean_squared_error(y_test,r_forest_predictions))

#Plotting residuals to check if they are normally distributed
sns.displot((y_test-r_forest_predictions),bins=50, kde=True)
plt.show()

r_forest_params = {'max_depth': [50,100,1000], 'min_samples_split': [50,100,1000], 'n_estimators':[50,100,1000]}

#Random forest after hyperparameter tuning
r_forest_grid = GridSearchCV(RandomForestRegressor(), r_forest_params, refit=True, verbose=3)
r_forest_grid.fit(x_train,y_train)

#Printing best params out of selected ones
print(r_forest_grid.best_params_)
r_forest_grid_predictions = grid.predict(x_test)
print('Predictions for model after hyperparameter tuning')
print('RMSE: ',metrics.root_mean_squared_error(y_test, r_forest_grid_predictions))
print('MAE: ',metrics.mean_absolute_error(y_test,r_forest_grid_predictions))
print('MSE: ',metrics.mean_squared_error(y_test,r_forest_grid_predictions))
sns.displot((y_test-r_forest_grid_predictions),bins=50, kde=True)
plt.title('Residuals of tuned Random Forest model')
plt.show()


