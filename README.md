# Random Forest

Team Members: Tristen Brewer, Nicholas Fenech, Janeel Abrahams

Title: Assessment 9 Group 3 (Sentinel) Machine Learning Presentation using Random Forest 

Project Description: Our goal is to  educate and provide insight on the use of the Random Forest algorithm. We will show three differet data sets varying progression of the algorithm from it's basic use to it's most optimal use. 

Our initial test will be the algorithm's most basic form, this will be done using the Cars93 dataset. Next we will add hyperparametrics to improve the accuracy of the model,this will be done using the California Housing dataset. Finally, we will have grid search paired with the hyperparameters to see how much the model can improve. For this we will use the 100,00 Cars dataset. We will use the ford.csv to train the model and the hyundi.csv to test the model

# Setting Up Your model

You will need the following import to replicate our results. 
### WARNING ALWAYS CHECK YOUR VERSION AS FUNCTIONS MAY BE DEPRECATED FOR OLDER VERSIONS

- `import pandas as pd`
- `from sklearn.model_selection import train_test_split`
- `from sklearn.ensemble import RandomForestRegressor`
- `from sklearn.model_selection import train_test_split,GridSearchCV`
- `from matplotlib import pyplot as plt`
- `import numpy as np`
- `import seaborn as sns`
- `from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,explained_variance_score`



Data Sources:  
  - 100,00 Used UK Cars [100,000 Used UK Cars](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?select=audi.csv)
  - Car93 [Cars93](https://www.kaggle.com/datasets/anand0427/cars93)
 -  California Housing[Fetch California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)


Files:
