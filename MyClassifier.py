from scipy.spatial import distance


def euc(a, b):
    return distance.euclidean(a, b)


class ScrappyKNN():
    """docstring for ScrappyKNN"""

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def predict(self, test_x):
        predictions = []
        for row in test_x:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dis = euc(row, self.train_x[0])
        best_index = 0
        for i in range(1, len(self.train_x)):
            dist = euc(row, self.train_x[i])
            if(dist < best_dis):
                best_dis = dist
                best_index = i
        return self.train_y[best_index]


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

# Here we will call our classifier
# from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()

my_classifier.fit(train_x, train_y)  # training with train features and train labels

predictions = my_classifier.predict(test_x)  # predicting for test features
print("My Predictions ", predictions)

from sklearn.metrics import accuracy_score
print("Accuracy is ", accuracy_score(test_y, predictions))  # comparing predicted values with actual test labels.
