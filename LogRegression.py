import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Create a classifier object to perform Logistic Regression tasks
lr = LogisticRegression()
# create a dataframe to store the data from the csv
data = pd.read_csv("data/neo_v2.csv")
# print the data so we can see it
print(data.head(5))

'''
# We will train our model on the training dataset and see how accurate the model is using the testing dataset.
# train_test_split will split up and randomize the data each time its run, meaning we will randomly select
80% of the data set to train with and randomly select 20% of the dataset to test with. Note that the testing 
and training data sets are disjoint.
'''
train, test = train_test_split(data, test_size=0.2)

'''
# Now we want to split up our training and testing datasets into their features and targets respectively.
Features are the predictive variables. In our case they are the est_diameter_min(2), est_diameter_max(3), 
relative_velocity(4), miss_distance(5), orbiting_body(6), sentry_object(7), and absolute_magnitude(8) columns. 
Note: Because orbiting_body(6) is not a numerical value we are not going to use it in the logistic regression model.
Targets are the response variables. In our case its the hazardous column (9)./
'''
training_features = train.iloc[:, [2, 3, 4, 5, 7, 8]]  # all rows and columns 3 - 9 in the training data set
training_targets = train["hazardous"]  # Training targets is a Series data type and not a DataFrame data type

testing_features = test.iloc[:, [2, 3, 4, 5, 7, 8]]  # all rows and columns 3 - 9 in the training data set
testing_targets = test["hazardous"]  # Training targets is a Series data type and not a DataFrame data type
print("Testing Features")
print(testing_features.head(5))
print("Testing Targets")
print(testing_targets.head(5))


'''
Now we are going to fit our logistic regression model and the see the fit score for the training and testing datasets.
Results show:
Training fit score:  0.9026118786811251
Testing fit score:  0.9029612505504183
Meaning the model got a score of 90.3/100 cases correctly labeled
'''
lr.fit(training_features, training_targets)
training_score = lr.score(training_features, training_targets)
print("Training fit score: ", training_score)


lr.fit(testing_features, testing_targets)
testing_score = lr.score(testing_features, testing_targets)
print("Testing fit score: ", testing_score)

'''
Log-Odds / Logits / Odds Ratio
Google this to find its meaning in logistic/ linear regression
'''
print(np.transpose(lr.coef_))

'''
Confusion Matrix
Gives us information on how many correct positives (cp) and correct negatives (cn) we found. As well as how many false 
positives (fp) and false negatives (fn) we found.
Ex: 
[[ cp fp]
 [ fn  cn]]
'''
training_conf_matrix = confusion_matrix(lr.predict(training_features), training_targets)
print("Training Confusion matrix")
print(training_conf_matrix)

testing_conf_matrix = confusion_matrix(lr.predict(testing_features), testing_targets)
print("Testing Confusion matrix")
print(testing_conf_matrix)
