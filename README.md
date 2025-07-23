<h1>Depression Prediction - Kaggle Competition</h1>

<img width="1735" height="153" alt="image" src="https://github.com/user-attachments/assets/9593440b-e969-4fb6-99b8-898a7499085c" />

<p>This project is a solution to the Kaggle competition <a href="https://www.kaggle.com/competitions/playground-series-s4e11/overview">Exploring Mental Health Data Challenge</a>. The goal is to build a machine learning model that can predict whether an individual is experiencing depression (<code>Depression = 1</code>) or not (<code>Depression = 0</code>) based on various features related to demographics, lifestyle, and mental health indicators.</p>

<!-- If you have an image, you can include it here -->
<!-- <img src="path_to_image" alt="Depression Prediction"> -->

<h2>File Explanations</h2>

<ul>
    <li><strong>main.ipynb</strong>: The Jupyter notebook containing the code for data loading, preprocessing, exploratory data analysis, modeling, and prediction.</li>
    <li><strong>train.csv</strong>: The artificial training dataset provided for the competition.</li>
    <li><strong>test.csv</strong>: The artificial test dataset provided for the competition.</li>
    <li><strong>original_dataset.csv</strong>: The original dataset.</li>
</ul>

<h2>Table of Contents</h2>

<ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#data-loading">Data Loading</a></li>
    <li><a href="#data-description">Data Description</a></li>
    <li><a href="#exploratory-data-analysis">Exploratory Data Analysis</a></li>
    <li><a href="#data-preprocessing">Data Preprocessing</a></li>
    <li><a href="#modeling">Modeling</a></li>
    <li><a href="#prediction-and-submission">Prediction and Submission</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
</ul>

<h2 id="overview">Overview</h2>

<p>In this project, we:</p>

<ul>
    <li>Loaded and preprocessed the dataset.</li>
    <li>Performed exploratory data analysis to understand the data.</li>
    <li>Engineered features and handled categorical variables.</li>
    <li>Used LightGBM with hyperparameter tuning using Optuna.</li>
    <li>Compared models trained on different datasets (original vs. artificial).</li>
    <li>Made predictions on the test set.</li>
    <li>Prepared submission files for Kaggle.</li>
</ul>

<h2 id="data-loading">Data Loading</h2>

<p>We imported necessary libraries and loaded the data into Pandas DataFrames.</p>

<pre><code>import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

dataset = pd.concat([train, test], ignore_index=True)
dataset.to_csv('original_dataset.csv', index=False)
</code></pre>

<h2 id="data-description">Data Description</h2>

<p>The dataset contains the following features:</p>

<ul>
    <li><strong>id</strong>: Unique identifier for each record.</li>
    <li><strong>Age</strong>: Age of the individual.</li>
    <li><strong>Gender</strong>: Gender of the individual.</li>
    <li><strong>City</strong>: City where the individual resides.</li>
    <li><strong>Profession</strong>: Profession of the individual.</li>
    <li><strong>Dietary Habits</strong>: Dietary habits (e.g., vegetarian, non-vegetarian).</li>
    <li><strong>Sleep Duration</strong>: Average sleep duration per night.</li>
    <li><strong>Financial Stress</strong>: Level of financial stress.</li>
    <li><strong>Family History of Mental Illness</strong>: Indicates if there's a family history of mental illness.</li>
    <li><strong>Job Satisfaction</strong>: Satisfaction level with the job.</li>
    <li><strong>Study Satisfaction</strong>: Satisfaction level with studies.</li>
    <li><strong>Work/Study Hours</strong>: Average work or study hours per day.</li>
    <li><strong>Depression</strong>: Target variable indicating depression status.</li>
</ul>

<h2 id="exploratory-data-analysis">Exploratory Data Analysis</h2>

<p>We performed exploratory data analysis to understand the distribution and relationships in the data.</p>

<ul>
    <li>Checked for differences between train and test sets.</li>
    <li>Analyzed correlations between numerical features.</li>
    <li>Visualized the distribution of the target variable.</li>
    <li>Explored key features like <strong>Age</strong>, <strong>Financial Stress</strong>, and <strong>Work/Study Hours</strong>.</li>
</ul>

<h2 id="data-preprocessing">Data Preprocessing</h2>

<p>We performed preprocessing steps:</p>

<ul>
    <li>Dropped irrelevant columns like <code>id</code>.</li>
    <li>Handled missing values.</li>
    <li>Encoded categorical variables using One-Hot Encoding.</li>
    <li>Transformed numerical features where necessary.</li>
</ul>

<h2 id="modeling">Modeling</h2>

<p>We used LightGBM (<code>LGBMClassifier</code>) for modeling and performed hyperparameter tuning using Optuna.</p>

<pre><code>from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Define features and target
X = dataset.drop(['Depression'], axis=1)
y = dataset['Depression']

# Preprocessing pipelines would be defined here (omitted for brevity)

# Hyperparameter tuning with Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 0,
    }
    model = LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accuracies = []
    
    for train_idx, valid_idx in skf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_valid_fold)
        accuracies.append(accuracy_score(y_valid_fold, y_pred))
    
    return np.mean(accuracies)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params

# Training the final model
model = LGBMClassifier(**best_params)
model.fit(X, y)
</code></pre>

<p>We compared models trained on different datasets (original vs. artificial) to evaluate performance.</p>

<h2 id="prediction-and-submission">Prediction and Submission</h2>

<p>We made predictions on the test set and prepared the submission files.</p>

<h2 id="conclusion">Conclusion</h2>

<p>We built machine learning models to predict depression status using LightGBM, with hyperparameter tuning. Comparing models trained on different datasets helped understand the impact of data augmentation.</p>

<p><strong>Key Takeaways:</strong></p>

<ul>
    <li>Younger individuals showed a higher likelihood of depression.</li>
    <li>Financial stress and work/study hours are significant factors.</li>
</ul>

<h2 id="acknowledgments">Acknowledgments</h2>

<p>We thank the Kaggle community and the competition organizers for providing the dataset and platform to enhance our machine learning skills.</p>
