from sklearn import datasets
iris = datasets.load_iris()

# f(x) = y

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.5)

# First way of creating classifier
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# Second way of creating classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(train_x, train_y)  # training with train features and train labels

predictions = my_classifier.predict(test_x)  # predicting for test features
print("My Predictions ", predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_y, predictions))  # comparing predicted values with actual test labels.
