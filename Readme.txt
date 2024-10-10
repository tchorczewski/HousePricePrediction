Housing Prices Prediction
Project Overview
This project uses machine learning models to predict housing prices based on various features in the California housing dataset. The primary goal is to explore different regression models and evaluate their performance.

Dataset:
The dataset used in this project is from the California Housing Prices and includes information like:

Median house values
Number of rooms, bedrooms, and households
Population in each block
Median Income
Proximity to the ocean
Project Steps
Data Exploration:

Initial overview of the data, including the structure, missing values, and basic statistical summaries.
Visualization of missing data and exploration of relationships between features using heatmaps, pairplots, and boxplots.
Data Preprocessing:

Filling missing data: Missing values in the total_bedrooms column were imputed based on the median values grouped by ocean_proximity.
Categorical to numerical conversion: The ocean_proximity column was transformed into dummy variables.
Skewed data transformation: Left-skewed numerical columns (e.g., total_rooms, total_bedrooms, population) were transformed using logarithmic scaling.
Data Scaling: Numerical columns (except for longitude and latitude) were standardized using StandardScaler to ensure all features are on the same scale.
Model Training:

Linear Regression: A simple linear regression model was trained to predict the median_house_value.
Decision Tree Regressor: A decision tree model was trained and evaluated.
Random Forest Regressor: A random forest model was trained and evaluated.
Model Evaluation:

The models were evaluated using:
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Mean Squared Error (MSE)

Performance Summary:

Linear Regression:
RMSE: 0.327
MAE: 0.243
MSE: 0.107
R^2: 0.672

Decision Tree Regressor:
RMSE: 0.422
MAE: 0.308
MSE: 0.178

Random Forest Regressor:
RMSE: 0.304
MAE: 0.223
MSE: 0.092

Conclusion:
The Random Forest Regressor performed the best, followed by Linear Regression. Decision Tree Regressor had the worst performance due to overfitting on the training data.
