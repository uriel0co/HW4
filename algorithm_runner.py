from sklearn.neighbors import KNeighborsClassifier, NearestCentroid


class AlgorithmRunner:
    def __init__(self, algorithm, k):
        if algorithm == "knn":
            self.algorithm = KNeighborsClassifier(n_neighbors=k)
        if algorithm == "Rocchio":
            self.algorithm = NearestCentroid()

    def fit(self, train_set, label_set):
        # this function train the algorithm on the train set.
        self.algorithm.fit(train_set, label_set)

    def predict(self, test_set):
        # this function predict the label of the samples in the test set.
        return self.algorithm.predict(test_set)

    def accuracy(self, predicted_labels,label_test):
        # this function calculate the precision, recall and accuracy of the algorithm and return those values.
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for predicted, true_label in zip(predicted_labels, label_test):
            if predicted == true_label:
                if predicted == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if predicted == 1:
                    fp += 1
                else:
                    fn += 1
        precision = tp/(fp + tp)
        recall = tp/(tp + fn)
        accuracy = (tp + tn)/len(label_test)
        return [precision, recall, accuracy]