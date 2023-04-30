# Random Forest

Team Members: Tristen Brewer, Nicholas Fenech, Janeel Abrahams

Title: Assessment 9 Group 3 (Sentinel) Machine Learning Presentation using Random Forest 

Project Description: Our goal is to  educate and provide insight on the use of the Random Forest algorithm. We will show three differet data sets varying progression of the algorithm from it's basic use to it's most optimal use. 

Our initial test will be the algorithm's most basic form, this will be done using the Cars93 dataset. Next we will add hyperparametrics to improve the accuracy of the model,this will be done using the California Housing dataset. Finally, we will have grid search paired with the hyperparameters to see how much the model can improve. For this we will use the 100,00 Cars dataset. We will use the ford.csv to train the model and the hyundi.csv to test the model


# Contents
- Jupyter Notebooks showing example of the model in practice
- PowerPoint presentation providing more details about Random Forest
- PDF of powerpoint slides

# Setting Up Your model
#### Be sure to correct your path depending on the data/Data folder used.

You will need the following imports to replicate our results. 
### WARNING:Always check your version as functions maybe deprecated 

- `import pandas as pd`
- `from sklearn.model_selection import train_test_split`
- `from sklearn.ensemble import RandomForestRegressor`
- `from sklearn.model_selection import train_test_split,GridSearchCV`
- `from matplotlib import pyplot as plt`
- `import numpy as np`
- `import seaborn as sns`
- `from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,explained_variance_score`

## General Structure
- Import the data you want to work on
- Do a bit od EDA
      - Are there nulls in your dataset
            - Replace?
            - Drop?
            - Fill?
      - What is the type of date in your column?
          - Will there be any conflicting data types
          - Will you need to create dummies?
      - Are the focus column the same size?
- Split your data into training and test/validation
      - `X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = decimal percentage of the data that you want to train on)`
     
- Create your model
      -  ` model = RandomForestRegressor(desired parameter)`
- Fit your model
      - `model = model.fit(X_train, y_train)`
- Score your model
      - `model = model.score(X_train, y_train)`
        
##### Repeat all the steps with the exception of fitting for your test dataset.

#### Optional metrics include:
- mean_squared_error
- mean_absolute_error
- r2_score (gives similar result of scored model)
- explained_variance_score (gives similar result of scored model)


##### If interested in doing what was done in the project explore the notebooks to see our exact steps.






Data Sources:  
 - 100,00 Used UK Cars [100,000 Used UK Cars](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?select=audi.csv)
 - Car93 [Cars93](https://www.kaggle.com/datasets/anand0427/cars93)
 -  California Housing [Fetch California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)


