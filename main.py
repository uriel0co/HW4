import sys
from algorithm_runner import AlgorithmRunner
from data import *


def main(argv):
    # movies = Data(argv[1])
    # movies.preprocess(['content_rating', 'movie_imdb_link', 'plot_keywords'], {})
    # print("Question 1:")
    # for al in ["KNN", "Rocchio"]:
    #     precision_recall_accuracy = [0, 0, 0]
    #     if al == "KNN":
    #         algorithm = AlgorithmRunner("knn", 10)
    #     else:
    #         algorithm = AlgorithmRunner("Rocchio", 10)
    #
    #     for train, test in movies.split_to_k_folds():
    #         train_set = movies.df.iloc[train, 0:-1].values
    #         test_set = movies.df.iloc[test, 0:-1].values
    #         label_train = movies.df.iloc[train, -1].values
    #         label_test = movies.df.iloc[test, -1].values
    #         algorithm.fit(train_set, label_train)
    #         predicted_labels = algorithm.predict(test_set)
    #         statistics = algorithm.accuracy(predicted_labels, label_test)
    #         for i in range(3):
    #             precision_recall_accuracy[i] += statistics[i]
    #     print("{} classifier: {},{},{}".format(al, precision_recall_accuracy[0]/5, precision_recall_accuracy[1]/5,
    #                                            precision_recall_accuracy[2]/5))
    # print()
    print("Question 2:")
    weights_vector = {'budget':3, 'language':3, 'num_voted_users':3, 'facenumber_in_poster':0.7, 'gross':3}
    movies = Data(argv[1])
    movies.preprocess(['content_rating', 'movie_imdb_link', 'plot_keywords', 'aspect_ratio', 'actor_1_facebook_likes',
                       'actor_2_facebook_likes', 'actor_3_facebook_likes'], weights_vector)
    for al in ["KNN", "Rocchio"]:
        precision_recall_accuracy = [0, 0, 0]
        if al == "KNN":
            algorithm = AlgorithmRunner("knn", 11)
        else:
            algorithm = AlgorithmRunner("Rocchio", 11)

        for train, test in movies.split_to_k_folds():
            train_set = movies.df.iloc[train, 0:-1].values
            test_set = movies.df.iloc[test, 0:-1].values
            label_train = movies.df.iloc[train, -1].values
            label_test = movies.df.iloc[test, -1].values
            algorithm.fit(train_set, label_train)
            predicted_labels = algorithm.predict(test_set)
            statistics = algorithm.accuracy(predicted_labels, label_test)
            for i in range(3):
                precision_recall_accuracy[i] += statistics[i]
        print("{} classifier: {}".format(al, precision_recall_accuracy[2] / 5))



if __name__ == "__main__":
    main(sys.argv)