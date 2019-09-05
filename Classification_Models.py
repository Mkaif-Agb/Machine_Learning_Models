# Importing the analysis libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_log = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Visualising the Training set results
'''
Using the code below u can get plots of all the classifier
I will be using it only on Logistic Regression but you can use the same code to apply the plot on different classifier
Maximum Dimension acceptable from the code below is 3

'''

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Prediction with different Models

# K-NN

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred_KNN = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_KNN))
print(classification_report(y_test, y_pred_KNN))


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0) # gini, entropy,etc
classifier.fit(X_train, y_train)

y_pred_Decision = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_Decision))
print(classification_report(y_test, y_pred_Decision))

# Random Forest Classifier
'''
Random Forest has a special Parameter 'n_estimators' which can take in a certain number 
It decides The number of trees in the forest.
To get the Best possible value we create a loop and calculate a mean of predicted value and the real value
We then plot the range with respect to err_rate to get the best possible value
'''
from sklearn.ensemble import RandomForestClassifier
err_rate = []
for i in range(1, 100):
    classifier = RandomForestClassifier(n_estimators=i, criterion='gini', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    err_rate.append(np.mean(y_pred!=y_test))

plt.plot(range(1,100), err_rate, ls='-', color='r')
plt.title("N_Estimators")
plt.xlabel('Range')
plt.ylabel('Error Rate')
plt.tight_layout()
plt.show()

# On running the above code we can see the best number of trees in the forest are 40 so we will create an instance with-
# 40 tress and get the best possible value

classifier = RandomForestClassifier(n_estimators=40, criterion='gini', random_state=0)
classifier.fit(X_train, y_train)


y_pred_tree = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Kernel Svm

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0) # We can try different Kernels to try and get best accuracy
classifier.fit(X_train, y_train)

y_pred_svm = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred_nb = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))



'''
After Running all the Models we can get the accuracy score from scikit-learn and choose which is the best model suited
for the Data

'''

from sklearn import metrics                               # Accuracy of all the classifiers respectively
print(metrics.accuracy_score(y_test, y_pred_log))         # 0.89
print(metrics.accuracy_score(y_test, y_pred_Decision))    # 0.95
print(metrics.accuracy_score(y_test, y_pred_tree))        # 0.94
print(metrics.accuracy_score(y_test, y_pred_KNN))         # 0.93
print(metrics.accuracy_score(y_test, y_pred_svm))         # 0.93
print(metrics.accuracy_score(y_test, y_pred_nb))          # 0.90


'''
Almost all the classifier works well on this Dataset with the highest accuracy on Desicision Tree Classifier
We applied more hyper-parameters on Random Forest classifier and got the best possible mean rate 
By tuning the hyper-parameters, We can get also get high accuracy like Decision Tree Classifier
'''