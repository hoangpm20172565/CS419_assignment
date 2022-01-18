import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":

    # load the data
    filePath = "data/svmTuningData.dat"
    file = open(filePath, 'r')
    allData = np.loadtxt(file, delimiter=',')

    X = allData[:, :2]
    y = allData[:, 2]

    # tuning
    tuned_parameters = [
        {"kernel": ["rbf"],
         "gamma": [1e-4, 1e-3, 0.01, 0.1, 1],
         "C": [0.1, 0.3, 0.6, 1, 3, 6, 10, 30 , 60 , 100, 300, 600, 1000, 3000]},

    ]

    score = 'accuracy'

    print(f"# Tuning hyper-parameters for {score}")
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, scoring=score)
    clf.fit(X, y)

    print("Best parameters set found:")
    print()
    print(clf.best_params_)
    print()
    print(f'Estimated accuracy: {clf.best_score_}')



    print("")
    print("Testing the SVMs...")

    h = .02  # step size in the mesh

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # get predictions for both my model and true model
    myPredictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    myPredictions = myPredictions.reshape(xx.shape)

    # plot my results
    plt.pcolormesh(xx, yy, myPredictions, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)  # Plot the training points
    plt.title(f"SVM with Optimal Gaussian Kernel  ({clf.best_params_})")
    plt.axis('tight')

    plt.show()