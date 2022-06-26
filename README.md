# Machine Learning using Nasa's Nearest Earth Objects
## Theory
> In this project we will be classifying whether an object is a threat to earth or not. We will be training and
> developing a machine learning model to perform this task.
> 
> "In machine learning, classification refers to a predictive modeling problem where a class label is predicted 
> for a given example of input data." (1)
> 
> In this project we will be utilizing **_Binary Classification_** which refers to those classification tasks that have two class labels.
> It is common to model a binary classification task with a model that predicts a Bernoulli probability distribution for each example.
> 
> The Bernoulli distribution is a discrete probability distribution that covers a case where an event will have a binary outcome as either a 0 or 1. For classification, this means that the model predicts a probability of an example belonging to class 1, or the abnormal state.
>
> Popular algorithms that can be used for binary classification include:
> - Logistic Regression
> - k-Nearest Neighbors
> - Decision Trees
> - Support Vector Machine
> - Naive Bayes



## Installing Requirements
> Run the following commands to pull the code to your local machine and install the dependency packages:
> 1. mkdir name
> 2. git clone https://github.com/Computer-Talkers/nasa-tracked-objects.git
> 3. python -m venv venv
> 4. source venv/bin/activate
> 5. pip install -r requirements.txt
> 
> Replace name in step 1 with what ever you want to name the directory that stores the code locally on your machine.


## Sources:
> 1. https://machinelearningmastery.com/types-of-classification-in-machine-learning/
> 2. https://scikit-learn.org/stable/