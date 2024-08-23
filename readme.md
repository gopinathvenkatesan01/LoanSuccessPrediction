# Loan Attraction Prediction

<p align="center">
  <img src="https://github.com/user-attachments/assets/59f79a9a-525e-4a5b-a0be-1a81846e765c" alt="Project WorkFlow" width="800" height="480">
 </p>

**Introduction**

The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts whether a customer is likely to be attracted to a loan offer. This predictive model, based on historical financial telemarketing data, aims to help financial companies efficiently identify and target potential customers who are most likely to respond positively to loan promotions.


**Table of Contents**

1. Key Technologies and Skills
2. Installation
3. Usage
4. Features
5. Contributing
6. License
7. Contact

**Key Technologies and Skills**
- Python
- Pandas
- Streamlit
- Plotly
- GridSearch CV
- SkLearn
- Pickle
- 
**Installation**

To run this project, you need to install the following packages:

```python
pip install pandas
pip install streamlit
pip install plotly
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install seaborn
```
**Usage**

To use this project, follow these steps:

1. Clone the repository: ```git clone https://github.com/gopinathvenkatesan01/SingaporeFlatePricePrediction```
2. Move Cursor to my app ```cd myapp```
3. Run the Streamlit app: ```streamlit run app.py```
4. Access the app in your browser at ```http://localhost:8501```

**Features**

**Data Preprocessing**
- **Dataset Overview :** The dataset consists of 45,211 rows and 11 columns related to a marketing campaign. Each row represents an individual's interaction with the campaign, and the goal is to predict the outcome (Outcome column) based on features such as age, job, marital status, education, and others.
- **Data Understanding :** Before embarking on modeling, it's essential to thoroughly comprehend your dataset. Begin by categorizing the variables, distinguishing between continuous and categorical ones, and examining their distributions.
- **Handling Null Values:** In our dataset, addressing missing values is critical. Depending on the nature of the data and the specific feature, we may opt for imputation methods such as mean, median, or mode to handle these null values effectively.
- **Encoding and Data Type Conversion:** When preparing categorical features for modeling, we utilize ordinal encoding. This method converts categorical values into numerical representations that reflect their inherent order and relationship with the target variable. Additionally, it's crucial to ensure that all data types are appropriately converted to meet the modeling requirements.
- **Skewness - Feature Scaling:** Skewness is a common challenge in datasets. Identifying skewness in the data is essential, and appropriate data transformations must be applied to mitigate it. One widely-used method is the log transformation, which is particularly effective in addressing high skewness in continuous variables. This transformation helps achieve a more balanced and normally-distributed dataset, which is often a prerequisite for many machine learning algorithms.
- **Outliers Handling:** Outliers have the potential to greatly influence model accuracy. To address outliers in our dataset, we employ the Interquartile Range (IQR) method. This approach entails identifying data points that lie outside the boundaries defined by the IQR and subsequently adjusting them to values that better align with the majority of the data. By applying this technique, we aim to enhance the robustness and reliability of our model predictions.
- **Model Building with Grid Search :** 
   - **Model Selection :**  For binary classification tasks, suitable models include XGBoost, RandomForestClassifier, and LogisticRegression. These models are commonly used due to their effectiveness in handling various types of data and their ability to manage overfitting through hyperparameter tuning.
   - **Train-Test Split :** The dataset is split into training and testing sets to evaluate the model's performance. The training set is used to train the model, while the testing set is reserved for assessing how well the model generalizes to unseen data. This step is crucial to avoid overfitting and to ensure that the model performs well on new data.
   - **Resampling with SMOTETomek :** In cases where the dataset is imbalanced (i.e., the number of instances in each class is not equal), SMOTETomek is employed to balance the classes in the training set. SMOTETomek combines the SMOTE (Synthetic Minority Over-sampling Technique) and Tomek links methods to improve the balance between classes, which is critical for training a robust and fair model.
   - **Grid Search for Hyperparameter Tuning :** Grid search is used to find the best hyperparameters for the chosen model. By exploring a predefined grid of hyperparameter values, grid search identifies the combination that optimizes the model's performance. This ensures that the model is finely tuned and performs optimally, leading to better predictions on the testing set.
    ```
        from sklearn.model_selection import GridSearchCV
        from xgboost import XGBClassifier

        param_grid = {
            'max_depth': [6, 9, 12],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
        }

        model = XGBClassifier()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
    ``` 
- **Model Serialization with Pickle :** Saving the Model and Scaler: Use pickle to save the trained model and scaler to disk. This allows you to load and use the model in your Streamlit application or any other deployment environment.
   ```
    import pickle
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    pickle.dump(best_model, open('xgb_model.pkl', 'wb'))
    ```
  

**Deployed in Streamlit**
  - **LoanAttractionPrediction:** [Loan Attraction Prediction](https://loanattractionpredictor.streamlit.app/)


**Results**

The project provides significant value to financial institutions by enabling them to identify and target customers who are most likely to respond positively to loan offers. The user-friendly web application allows marketing teams to efficiently filter potential customers, optimizing their outreach efforts and improving conversion rates. Additionally, the project highlights the practical application of machine learning in the financial sector, demonstrating how predictive models can be effectively integrated into decision-making processes to enhance business outcomes.

**Contributing**

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request.

**License**

This project is licensed under the MIT License. Please review the LICENSE file for more details.

**Contact**

📧 Email: gopinathvenakatesan01@gmail.com

🌐 LinkedIn: [linkedin.com/in/gopinath-venkatesan-9707022a7](https://www.linkedin.com/in/gopinath-venkatesan-9707022a7/)

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.
