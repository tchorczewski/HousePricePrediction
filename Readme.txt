This project explores the use of machine learning models to predict housing prices based on various features from the California housing dataset. The primary goal is to experiment with different regression models, evaluate their performance, and improve predictive accuracy through hyperparameter tuning.

Dataset:
The dataset used is the California Housing Prices dataset from Kaggle platform, which includes features such as:

-Median house values
-Number of rooms, bedrooms, and households
-Population per block
-Median income
-Proximity to the ocean

Project Steps:
1. Data Exploration
Overview of the dataset, checking its structure, missing values, and summary statistics.
Visual exploration using heatmaps, pairplots, and boxplots to understand relationships between features and detect any anomalies.
2. Data Preprocessing
Filling Missing Data: Missing values in the total_bedrooms column were filled using the median values grouped by ocean_proximity.
Converting Categorical Data: The ocean_proximity column was transformed into dummy variables.
Transforming Skewed Data: Logarithmic transformation was applied to left-skewed numerical features (total_rooms, total_bedrooms, population, etc.).
Data Scaling: Numerical columns, excluding longitude and latitude, were standardized using StandardScaler to ensure consistent scaling across features.
3. Model Training
Linear Regression: A simple linear regression model was trained to predict the median_house_value.
Decision Tree Regressor: A decision tree was trained and further fine-tuned using hyperparameter optimization.
Random Forest Regressor: A random forest model was trained and optimized through hyperparameter tuning to increase prediction accuracy.
4. Model Evaluation
Each model was evaluated using:

-Root Mean Squared Error (RMSE)
-Mean Absolute Error (MAE)
-Mean Squared Error (MSE)
-R² Score

Performance Summary:

Model	                       RMSE	   MAE	    MSE	  R² Score
Linear Regression	         0.3271	0.2432	0.1070	0.6720
Decision Tree (Pre-tuning)	0.4211	0.3074	0.1773	0.4562
Decision Tree (Post-tuning)	0.3260	0.2451	0.1063	0.6741
Random Forest (Pre-tuning)	0.3039	0.2226	0.0924	0.7167
Random Forest (Post-tuning)	0.3260	0.2451	0.1063	0.6741

Conclusion:
The Random Forest Regressor exhibited the best performance in terms of RMSE, MSE, and R² score, outperforming both the Linear Regression and Decision Tree models.

Although the Decision Tree Regressor initially performed poorly due to overfitting, its performance significantly improved after hyperparameter tuning, making it comparable to the other models. However, the Random Forest model maintained a slight edge in predictive power. The Linear Regression model, though simpler, also delivered solid results, proving that even basic models can be effective depending on the dataset.

Future work could explore other ensemble methods (e.g., gradient boosting) and deeper feature engineering to further enhance prediction accuracy.
