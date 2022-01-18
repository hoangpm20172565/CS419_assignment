'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score


class Learning_Curve_data:
    def __init__(self):
        self.percentage = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
        self.data = {}
        for p in self.percentage:
            self.data[p] = []
        self.curve = []
        self.std = []

    def summary(self):
        for p in self.percentage:
            a = np.array(self.data[p])
            self.curve.append(a.mean())
            self.std.append(a.std())


def generate_learning_curve(dict_of_lcdata):
    x = np.arange(10, 110, 10)
    for key, item in dict_of_lcdata.items():
        item.summary()
        mean = np.array(item.curve)
        print(f'{key}\'s mean: {mean}')
        std = np.array(item.std)
        print(f'{key}\'s std: {std}')
        plt.fill_between(x, (mean+std)*100, (mean-std)*100, alpha=0.2)
        plt.plot(x, mean*100, label=key)

    plt.legend()
    plt.grid(True)
    plt.title('Learning Curve for Decision Trees')
    plt.xlabel('Percentage of training data (%)')
    plt.ylabel('Accuracy (%)')
    plt.xlim(10,100)
    plt.ylim(0,100)
    plt.show()



def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    trees = ['unlimited', 'stump', 'dt3', 'dt4', 'dt5']
    lc = {}
    for type in trees:
        lc[type] = Learning_Curve_data()

    meantree = []
    meanstump = []
    meandt3 = []

    clf = tree.DecisionTreeClassifier()
    stump = tree.DecisionTreeClassifier(max_depth=1)
    dt3 = tree.DecisionTreeClassifier(max_depth=3)
    dt4 = tree.DecisionTreeClassifier(max_depth=4)
    dt5 = tree.DecisionTreeClassifier(max_depth=5)

    idx = np.arange(n)
    print(f'Number of data: {n}')
    np.random.seed(13)
    
    # 100 loops
    for _ in range(100):
        print(f'Loop number {_}')
        # shuffle the data
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # split the data
        X_10fold = np.array_split(X, 10)
        y_10fold = np.array_split(y, 10)

        # 10 fold
        for i in range(10):
            # pick out 1 fold for testing
            Xtest = X_10fold[i]
            ytest = y_10fold[i]

            # use other 9 folds for training
            X_train = [X_10fold[j] for j in range(10) if j!=i]  # train on first 9 folds
            y_train = [y_10fold[j] for j in range(10) if j!=i]
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)

            # again split the training data into 10 parts for generate the leaning curve
            X_split = np.array_split(X_train, 10)
            y_split = np.array_split(y_train, 10)

            # increase the number of training datas from 10% to 100%
            for j in range(10):
                p = j+1
                Xtrain = [X_split[k] for k in range(p)]  # increase through each loop
                ytrain = [y_split[k] for k in range(p)]
                Xtrain = np.concatenate(Xtrain)
                # print(f'{i} - {j}: {Xtrain.shape}')
                ytrain = np.concatenate(ytrain)

                # train the decision tree
                clf = clf.fit(Xtrain,ytrain)
                stump = stump.fit(Xtrain,ytrain)
                dt3 = dt3.fit(Xtrain,ytrain)
                dt4 = dt4.fit(Xtrain,ytrain)
                dt5 = dt5.fit(Xtrain,ytrain)

                # output predictions on the remaining data
                predictions = []
                predictions.append(clf.predict(Xtest))
                predictions.append(stump.predict(Xtest))
                predictions.append(dt3.predict(Xtest))
                predictions.append(dt4.predict(Xtest))
                predictions.append(dt5.predict(Xtest))

                for type, pred in zip(trees, predictions):
                    lc[type].data[str(p*10)].append(accuracy_score(ytest, pred))
                    # print(f'Saving {accuracy_score(ytest, pred)} to {type}: {p*10}')

                # compute the training accuracy of the model
                if j==9:
                    meantree.append(accuracy_score(ytest, predictions[0]))
                    meanstump.append(accuracy_score(ytest, predictions[1]))
                    meandt3.append(accuracy_score(ytest, predictions[2]))


    meantree = np.array(meantree)
    meanstump = np.array(meanstump)
    meandt3 = np.array(meandt3)
    # print(meantree)
    # TODO: update these statistics based on the results of your experiment
    meanDecisionTreeAccuracy = meantree.mean()
    stddevDecisionTreeAccuracy = meantree.std()
    meanDecisionStumpAccuracy = meanstump.mean()
    stddevDecisionStumpAccuracy = meanstump.std()
    meanDT3Accuracy = meandt3.mean()
    stddevDT3Accuracy = meandt3.std()

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy

    generate_learning_curve(lc)
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
    print("Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
    print("3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")
# ...to HERE.
