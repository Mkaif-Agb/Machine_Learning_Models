# Machine_Learning_Models
Classification algorithms do what the name suggests i.e. they train models to predict what class some object belongs to. A very common application is image classification. Given some photo, what is it? It is the success of solving that kind of problem with sophisticated deep neural networks running on GPU's that caused the big resurgence of interest in machine learning a few years ago.

Using Different Machine Learning Models to get the best possible accuracy on the Dataset.In this Project We will see the implementation of various Machine Learning models such as
Logistic Regression, Decision Tree, Random Forest, Naive bayes, K-Nearest Neighbor and SVM(Support-Vector-Machine)

# Logistic Regression
Logistic Regression is an algorithm that is relatively simple and powerful for deciding between two classes, i.e. it's a binary classifier. It basically gives a function that is a boundary between two different classes. It can be extended to handle more than two classes by a method referred to as "one-vs-all" (multinomial logistic regression or softmax regression) which is really a collection of binary classifiers that just picks out the most likely class by looking at each class individually verses everything else and then picks the class that has the highest probability.

![Logistic Regression](http://storage.ning.com/topology/rest/1.0/file/get/2808358994?profile=original)

# Decision Tree
The decision making tree is one of the better known decision making techniques, probably due to its inherent ease in visually communicating a choice, or set of choices, along with their associated uncertainties and outcomes. Their simple structure enables use in a broad range of applications. They can be drawn by hand to help quickly outline and communicate the critical elements in a decision. Alternatively, a decision tree's simple logical structure enables it to be used to address complex multiple decision scenarios and problems with the aid of computers.
![Decision Tree Example](https://miro.medium.com/max/2000/1*WerHJ14JQAd3j8ASaVjAhw.jpeg)

# Random Forest

Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large. The generalization error of a forest of tree classifiers depends on the strength of the individual trees in the forest and the correlation between them. Using a random selection of features to split each node yields error rates that compare favorably to Adaboost (Freund and Schapire[1996]), but are more robust with respect to noise. Internal estimates monitor error, strength, and correlation and these are used to show the response to increasing the number of features used in the splitting. Internal estimates are also used to measure variable importance. These ideas are also applicable to regression.
![Random Forest Example](https://miro.medium.com/max/1170/1*58f1CZ8M4il0OZYg2oRN4w.png)

# Naive bayes

It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.
![Naive bayes Example](https://s3.ap-south-1.amazonaws.com/techleer/204.png)

# K-Nearest Neighbor
The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.However, it is more widely used in classification problems in the industry. To evaluate any technique we generally look at 3 important aspects:
1. Ease to interpret output
2. Calculation time
3. Predictive Power
![K-Nearest Neighbor](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final1_ibdm8a.png)

# Support Vector Machine
A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two classes very well (look at the below snapshot).
![Support Vector Machine](https://miro.medium.com/max/1200/1*06GSco3ItM3gwW2scY6Tmg.png)

