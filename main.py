import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

housing = pd.read_csv('housing.csv')
print(housing.info())
print(housing.describe())
print(housing.head())
#Data Exploration
sns.heatmap(housing.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null values visualization')
plt.show()