import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

housing = pd.read_csv('housing.csv')
print(housing.info())
print(housing.describe())
print(housing.head())
print(housing['ocean_proximity'].value_counts())

#Data Exploration
#Missing values heatmap
sns.heatmap(housing.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null values visualization')
plt.show()

#Number of missing values per column
print(housing.isna().sum())
print(housing.loc[housing['total_bedrooms'].isnull()])

#Data Exploration
#Missing values heatmap
sns.heatmap(housing.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null values visualization')
plt.show()

#Number of missing values per column
print(housing.isna().sum())
print(housing.loc[housing['total_bedrooms'].isnull()])

#Filling missing data
ocn_prox_data = housing.groupby('ocean_proximity')
missing_bedrooms = housing[housing['total_bedrooms'].isnull()]
sns.countplot(data=missing_bedrooms, x='ocean_proximity')
plt.title('Missing bedroom data vs ocean_proximity')
plt.show()

print(ocn_prox_data['total_bedrooms'].mean())
print(ocn_prox_data['total_bedrooms'].median())

sns.heatmap(housing.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null values visualization after filling')
plt.show()

#Median house value vs ocean proximity
sns.boxplot(data=housing, x='ocean_proximity', y='median_house_value')
plt.title('House value vs ocean proximity')
plt.show()

print(ocn_prox_data['median_house_value'].mean())
print(ocn_prox_data['median_house_value'].median())

#Median price vs ocean proximity
sns.boxplot(data=housing, x='ocean_proximity', y='median_income' )
plt.title('Income vs Ocean Proximity')
plt.show()

#Exploring correlation between data
sns.pairplot(housing, hue='ocean_proximity')
plt.title('Housing numerical data relations')
plt.show()

sns.heatmap(data=housing.corr(numeric_only=True), cmap='coolwarm')
plt.title('Correlation heatmap')
plt.show()

#Median Income vs Median house price
sns.lmplot(data=housing, x='median_income', y='median_house_value')
plt.title("Income vs house value")
plt.show()
