from sklearn import tree

# 1 is for smooth, 0 is for bumpy
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 for Apple, 1 for orange
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
# fit = find patterns in data
clf = clf.fit(features, labels)
print(clf.predict([[150, 0],[160,0],[180,1],[120,0],[120,1]]))

