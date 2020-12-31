import pandas as pd
import sklearn
import numpy as np


class Data:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.non_categorials = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                                'aspect_ratio', 'budget', 'cast_total_facebook_likes', 'director_facebook_likes',
                                'duration', 'facenumber_in_poster', 'gross', 'movie_facebook_likes',
                                'num_critic_for_reviews', 'num_user_for_reviews', 'num_voted_users', 'title_year']

    def preprocess(self, features, weights_vector):
        # this function delete the unwanted features, change the categorical features to dummy features and normalize
        # the non categorical features.
        self.df.drop(columns=features, axis=1, inplace=True)  # remove unwanted cols
        self.df.replace('', np.nan, inplace=True)  # replace empty element wits "NaN"
        self.df.dropna(inplace=True)  # drop NaNs
        self.df.drop_duplicates(subset=['movie_title'], keep='first',
                                inplace=True)  # drop movies with same title - keeps first

        # DEALS WITH CATEGORICAL FEATURES
        self.df['actor_3_name'] = self.df['actor_3_name'] + '|' + self.df['actor_2_name'] + '|' + self.df['actor_1_name']
        self.df.dropna(inplace=True)
        self.df.drop(columns=['actor_2_name', 'actor_1_name'], inplace=True)

        dummy_column = self.df["actor_3_name"].str.get_dummies()
        self.df.drop(columns="actor_3_name", inplace=True)
        self.df = pd.concat([self.df, dummy_column], axis=1)

        for key in ['color', 'director_name', 'genres', 'language', 'country']:
            if key == "genres":
                dummy_column = self.df["genres"].str.get_dummies()
                self.df.drop(columns="genres", inplace=True)
                if key in weights_vector.keys():
                    dummy_column *= weights_vector[key]
                self.df = pd.concat([self.df, dummy_column], axis=1)
            else:
                if key in weights_vector.keys():
                    dummy_column *= weights_vector[key]
                self.df = pd.get_dummies(self.df, columns=[key])
        for key in self.df.columns.values.tolist():
            if key not in ['color', 'director_name', 'genres', 'language', 'country']:
                if key in weights_vector.keys():
                    self.df[key] *= weights_vector[key]
            label = []
        for imbd_value in self.df['imdb_score']:
            if imbd_value >= 7:
                label.extend([1])
            else:
                label.extend([0])
        self.df["label"] = label
        self.df.drop(columns=["movie_title", 'imdb_score'], axis=1, inplace=True)
        #Normalization
        for col in self.non_categorials:
            if col in self.df.keys():
                self.df[col].update((self.df[col] - self.df[col].mean()) / np.std(self.df[col]))
        return self.df


    def split_to_k_folds(self):
        kf = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=None)
        return kf.split(self.df)