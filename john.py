import pandas as pd
import numpy as np
import os, wandb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

choice = input("Do you have the W&B API key?: (y or n): ")
choice = choice.lower()
if choice == 'y':
    print("Make sure you did 'pip install wandb' ")
    print("Going to run wandb login and you can paste your API key right in the ")
    os.system("wandb login")
    
    # Going to init the project with a new run 
    wandb.init(project="Nasa Tracked Objects", entity="computer-talkers")

# Create a classifier object to perform Logistic Regression tasks
lr = LogisticRegression()
# create a dataframe to store the data from the csv


raw_data = pd.read_csv("data/neo_v2.csv")
# print the data so we can see it
print("Here is a preview of the data raw data: ")
print(raw_data.head(5))

print("""
\nThe data cleaning protocol includes the following tasks:

1) Removing the non-numerical columns from the data.
2) Changing the True/False to 1/0 (True is 1 and False is 0).
3) Removing any features that may be correlated to another.
4) Removing class imbalance so that the data set has a 50-50 split of true and false values in the target.""")

# (1) There is only one numerical column here:
clean_data = raw_data.drop(["orbiting_body", "sentry_object", "id", "name"], axis=1)

# (2) Changing sentry_object and hazardous from True/False to 1/0
bool_columns = ["hazardous"]
clean_data[bool_columns] = clean_data[bool_columns].astype(int)

# (3) Removing any features that may be correlated to another.
clean_data = clean_data.drop(["est_diameter_min"], axis=1)

# (4) Removing class imbalance
# Separate majority and minority classes
hazardous_majority = clean_data[clean_data.hazardous == 0]
hazardous_minority = clean_data[clean_data.hazardous == 1]

# Downsample majority class
majority_downsampled = resample(hazardous_majority, replace=False, n_samples=8840)

# Combine minority class with downsampled majority class
clean_data = pd.concat([majority_downsampled, hazardous_minority])

print("Here is a preview of the cleaned data: ")
print(clean_data.head(5))


'''
# We will train our model on the training dataset and see how accurate the model is using the testing dataset.
# train_test_split will split up and randomize the data each time its run, meaning we will randomly select
80% of the data set to train with and randomly select 20% of the dataset to test with. Note that the testing 
and training data sets are disjoint.
'''
train, test = train_test_split(clean_data, test_size=0.2)

'''
# Now we want to split up our training and testing datasets into their features and targets respectively.
Features are the predictive variables. In our case they are the est_diameter_max(0), relative_velocity(1),
miss_distance(2), absolute_magnitude(3) columns. 
Targets are the response variables. In our case its the hazardous column (4)./

'''
training_features = train.iloc[:, [0, 1, 2, 3]]  # all rows and columns 1 - 3 in the training data set
training_targets = train["hazardous"]  # Training targets is a Series data type and not a DataFrame data type

testing_features = test.iloc[:, [0, 1, 2, 3]]  # all rows and columns 1 - 3 in the testing data set
testing_targets = test["hazardous"]  # Training targets is a Series data type and not a DataFrame data type

'''
Now we are going to fit our logistic regression model and then see the fit score for the training and testing datasets.
Results show:
Training fit score:  0.9026118786811251
Testing fit score:  0.9029612505504183
Meaning the model got a score of 90.3/100 cases correctly labeled
'''
lr.fit(training_features, training_targets)
training_score = lr.score(training_features, training_targets)
print(f"Training fit score: {training_score}\n")


lr.fit(testing_features, testing_targets)
testing_score = lr.score(testing_features, testing_targets)
print(f"Testing fit score: {testing_score}\n")

# Plotting regressor to wandb
if choice == 'y':
    # wandb.sklearn.plot_regressor(lr, training_features, training_targets, testing_features, testing_targets, model_name="Nasa Objects")
    # wandb.sklearn.plot_regressor(lr, training_targets, testing_targets, training_features, testing_features, model_name="Nasa Tracked Objects")
    wandb.sklearn.plot_learning_curve(lr, training_targets, training_targets)
    # wandb.sklearn.plot_class_proportions(training_features, testing_features)
'''
Log-Odds / Logits / Odds Ratio
Google this to find its meaning in logistic/ linear regression
'''
print("Log-Odds")
print(f"{np.transpose(lr.coef_)}\n")

'''
Confusion Matrix
Gives us information on how many correct positives (cp) and correct negatives (cn) we found. As well as how many false 
positives (fp) and false negatives (fn) we found.
Ex: 
[[ cp fp]
 [ fn  cn]]
'''
training_features_pred = lr.predict(training_features)
training_conf_matrix = confusion_matrix(training_targets, training_features_pred)

print("Training Confusion matrix")
print(f"{training_conf_matrix}\n")

testing_features_pred = lr.predict(testing_features)
testing_conf_matrix = confusion_matrix(testing_targets, testing_features_pred)
if choice == 'y':
    wandb.sklearn.plot_confusion_matrix(testing_targets, testing_features_pred)
    wandb.finish()
print("Testing Confusion matrix")
print(f"{testing_conf_matrix}\n")