from Dataset import get_stocks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from collections import Counter


#
#  Analysis predicts future stock values based on historical data
#
class Analysis:

    def __init__(self):
        self.dataset = None
        self.clf = None

    def load_data(self):
        dataset = get_stocks()
        print(dataset[1].target)
        self.dataset = dataset

    def train_model(self):

        # clf = VotingClassifier([("LinearSVM", SVC()),
        #                         ("knn", KNeighborsClassifier())])
        clf = SGDClassifier()

        for _ in [1,2]:
            for ds in self.dataset:
                x_train, x_test, y_train, y_test = train_test_split(ds.data[-365*4:], ds.target[-365*4:], random_state=1)
                clf.partial_fit(x_train, y_train, [0, 1])

        for ds in self.dataset:
            _, _, y_train, y_test = train_test_split(ds.data, ds.target, random_state=1)
            accuracy = clf.score(ds.data, ds.target)
            prediction = clf.predict(ds.data)
            print(ds.name)
            print("Accuracy:", accuracy)
            print("Prediction:", Counter(prediction))

        self.clf = clf


    def predict_stock(self):
        return 0


an = Analysis()

an.load_data()
an.train_model()
