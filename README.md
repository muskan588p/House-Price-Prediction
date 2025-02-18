# House-Price-Prediction

This project utilizes machine learning algorithms to predict house prices based on key features such as the number of bedrooms, square footage, location, and additional property characteristics. By analyzing historical housing data, the goal is to build a predictive model that can accurately estimate house prices. The project involves data preprocessing, feature selection, and applying various machine learning techniques, including regression models and decision trees, to identify the most accurate prediction model. The outcome provides valuable insights for real estate professionals, helping them to make informed decisions based on property attributes.

Overview

In this project, I used Python and machine learning libraries to develop a model that predicts the price of houses. The model was trained using historical housing data, and several preprocessing techniques like data cleaning, feature selection, and exploratory data analysis (EDA) were performed to enhance its accuracy.
Technologies Used

    Programming Languages: Python
    Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
    Data Visualization: Matplotlib, Seaborn
    Machine Learning Algorithms: Linear Regression, Random Forest, Decision Trees
    Environment: Jupyter Notebook

Dataset

The dataset used in this project is a collection of housing data that contains various features of houses (such as size, number of rooms, location, etc.) and their corresponding prices.

You can find the dataset in the data/ folder. Alternatively, you can use publicly available datasets such as the Kaggle House Prices dataset.

Model

    Preprocessing: Handled missing data, normalized features, and encoded categorical variables.
    Algorithms:
        Linear Regression
        Random Forest
        Decision Trees
    Evaluation Metrics: RMSE (Root Mean Squared Error), R² (Coefficient of Determination)

Results and Analysis

We evaluated multiple regression models to predict house prices based on features such as the number of bedrooms, square footage, and location. The models tested include Linear Regression, Decision Tree, and Neural Network.

    Linear Regression achieved the lowest Mean Absolute Error (MAE) of 14,843.53, making it the best-performing model.
    Decision Tree followed with an MAE of 20,748.03.
    Neural Network had the highest MAE of 41,809.11, indicating lower accuracy compared to the other models.

The MAE values indicate that Linear Regression performed well in predicting house prices, with errors averaging around 14,000, which is acceptable given the price distribution in the dataset (mean: 179,846.69).

Visualizations such as histograms, box plots, and violin plots were used to further understand the distribution of house prices and to assess the model's performance.

In conclusion, Linear Regression was the most reliable model for predicting house prices in this project.

Usage

    Load the dataset by running the house_price_prediction.ipynb notebook.
    Perform exploratory data analysis (EDA) to understand patterns and correlations in the data.
    Split the data into training and testing sets, then train the machine learning models.
    Evaluate the performance of the models using the appropriate metrics.
    Use the model to make predictions on new data.
