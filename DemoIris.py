import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
# metadata imported
print(iris.feature_names)  # this gives names of features
print(iris.target_names)  # this gives names of flowers
#  *****
# print(iris.data[0])  # iris.data is the data set
# print(iris.target[0])  # iris.target is the collection of flower names against each member in above data set

# to retrive all the flower names :
# for i in range(len(iris.target)):
# print("Example %d: Label %s , features %s" % (i, iris.target[i], iris.data[i]))


# for testing purpose we will remove some data points from dataset
test_idx = [0, 50, 100]  # i'll remove these 3 indexes from dataset

# trainig data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print("Ideal answer should be ", test_target)
print("algo's answer ", clf.predict(test_data))


# generate DecisionTree PDF
# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
