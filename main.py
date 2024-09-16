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

housing = pd.read_csv('housing.csv')
print(housing.info())
print(housing.describe())
print(housing.head())
print(housing['ocean_proximity'].value_counts())

#Data Exploration
#Missing values heatmap
sns.heatmap(housing.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null values visualization')
#plt.show()

#Number of missing values per column
print(housing.isna().sum())
print(housing.loc[housing['total_bedrooms'].isnull()])

#Filling missing data
ocn_prox_data = housing.groupby('ocean_proximity')
missing_bedrooms = housing[housing['total_bedrooms'].isnull()]
sns.countplot(data=missing_bedrooms, x='ocean_proximity')
plt.title('Missing bedroom data vs ocean_proximity')
#plt.show()

print(ocn_prox_data['total_bedrooms'].mean())
print(ocn_prox_data['total_bedrooms'].median())

housing['total_bedrooms'] = housing['total_bedrooms'].fillna(housing.groupby('ocean_proximity')['total_bedrooms']
                                                             .transform('median'))

sns.heatmap(housing.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null values visualization after filling')
#plt.show()

#Median house value vs ocean proximity
sns.boxplot(data=housing, x='ocean_proximity', y='median_house_value')
plt.title('House value vs ocean proximity')
#plt.show()

print(ocn_prox_data['median_house_value'].mean())
print(ocn_prox_data['median_house_value'].median())

#Exploring correlation between data
sns.pairplot(housing, hue='ocean_proximity')
plt.title('Housing numerical data relations')
#plt.show()

sns.heatmap(data=housing.corr(numeric_only=True), cmap='coolwarm')
plt.title('Correlation heatmap')
#plt.show()

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

#Training the model
linear_regression = LinearRegression()
print(linear_regression.fit(x_train,y_train))
predictions = linear_regression.predict(x_test)

#Plotting residuals to check if they are normally distributed
sns.displot((y_test-predictions),bins=50, kde=True)
#plt.show()

#Calculating performance metrics
print('Linear Regression Metrics')
print(metrics.root_mean_squared_error(y_test, predictions))
print(metrics.mean_absolute_error(y_test,predictions))
print(metrics.mean_squared_error(y_test,predictions))

#Decision Tree Regressor
d_tree = DecisionTreeRegressor()
d_tree.fit(x_train,y_train)
d_tree_predictions = d_tree.predict(x_test)
print('Decision Tree Regressor Metrics')
print(metrics.root_mean_squared_error(y_test, d_tree_predictions))
print(metrics.mean_absolute_error(y_test,d_tree_predictions))
print(metrics.mean_squared_error(y_test,d_tree_predictions))

#Random Forest Regressor
r_forest = RandomForestRegressor()
r_forest.fit(x_train,y_train)
r_forest_predictions = r_forest.predict(x_test)
print('Random Forest Regressor Metrics')
print(metrics.root_mean_squared_error(y_test, r_forest_predictions))
print(metrics.mean_absolute_error(y_test,r_forest_predictions))
print(metrics.mean_squared_error(y_test,r_forest_predictions))