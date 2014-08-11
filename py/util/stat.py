__author__ = 'msemeniuk'
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import ensemble

class FantasyStats(classmethod):
    def __init__(self, df):
        self.score_year=2013
        self.df=df
        self.df_train = pd.DataFrame()
        self.df_valid = pd.DataFrame()
        self.df_score = pd.DataFrame()
        self.clfr = ensemble.GradientBoostingRegressor()
        self.clfc = ensemble.GradientBoostingClassifier()

    def build_train_valid_score(self, pct_train):
        self.df.set_index('year', inplace=True, drop=False)
        self.df_score = self.df.ix[self.score_year]
        self.df.reset_index(inplace=True, drop=True)

        #Build Prediction Data
        #self.df.set_index('PLAYER', inplace=True, drop=False)
        #self.df_score.set_index('PLAYER', inplace=True, drop=False)

        #Define Train and Validation Datasets
        ix = shuffle(self.df.index)
        n_train = int(len(ix) * pct_train / 100)
        ix_t = ix[:n_train]
        ix_v = ix[n_train:]

        self.df_train = self.df.ix[ix_t]
        self.df_valid = self.df.ix[ix_v]

        self.df_train.set_index('PLAYER', inplace=True, drop=False)
        self.df_valid.set_index('PLAYER', inplace=True, drop=False)

    def get_df_train(self):
        return self.df_train[self.df_train.year < self.score_year]

    def get_df_valid(self):
        return self.df_valid[self.df_valid.year < self.score_year]

    def get_df_score(self):
        return self.df_score

    def get_gbr_result(self):
        return self.clfr

    def get_gbc_result(self):
        return self.clfc

    def train_gbr(self, dep_var, params, feature_vector):
        """
        Used for Gradient Boosted Regression, like predicted rank
        :param dep_var:
        :param params:
        :return:
        """
        X = self.df_train[(pd.isnull(self.df_train[dep_var]) == False)][feature_vector]
        Y = self.df_train[pd.isnull(self.df_train[dep_var]) == False][dep_var]
        X = X.astype(np.float32)

        self.clfr = ensemble.GradientBoostingRegressor(**params)

        self.clfr.fit(X, Y)

    def train_gbc(self, dep_var, params, feature_vector):
        """
        Used for Gradient Boosted Classifier, like predicted to be in top 10
        TODO: add a check that the dependent variable is really categorical (1/0)
        :param dep_var:
        :param params:
        :return:
        """
        X = self.df_train[(pd.isnull(self.df_train[dep_var]) == False)][feature_vector]
        Y = self.df_train[pd.isnull(self.df_train[dep_var]) == False][dep_var]
        X = X.astype(np.float32)

        self.clfc = ensemble.GradientBoostingClassifier(**params)
        self.clfc.fit(X, Y)



    def train_gbr_fulL_sample(self, dep_var, params, feature_vector):
        """
        Used for Gradient Boosted Regression, like predicted rank
        :param dep_var:
        :param params:
        :return:
        """
        df_full = pd.concat([self.df_train, self.df_valid])
        X = df_full[(pd.isnull(df_full[dep_var]) == False)][feature_vector]
        Y = df_full[pd.isnull(df_full[dep_var]) == False][dep_var]
        X = X.astype(np.float32)

        self.clfr = ensemble.GradientBoostingRegressor(**params)

        self.clfr.fit(X, Y)

    def train_gbc_full_sample(self, dep_var, params, feature_vector):
        """
        Used for Gradient Boosted Classifier, like predicted to be in top 10
        TODO: add a check that the dependent variable is really categorical (1/0)
        :param dep_var:
        :param params:
        :return:
        """
        df_full = pd.concat([self.df_train, self.df_valid])
        X = df_full[(pd.isnull(df_full[dep_var]) == False)][feature_vector]
        Y = df_full[pd.isnull(df_full[dep_var]) == False][dep_var]
        X = X.astype(np.float32)

        self.clfc = ensemble.GradientBoostingClassifier(**params)
        self.clfc.fit(X, Y)


    def feature_importance(self, clf, x_vars):
        feature_importance = clf.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        for i in sorted_idx:
            print np.array(x_vars)[i] + ' '* (20 - len(np.array(x_vars)[i])) + str(feature_importance[i])

    def score_train_and_validation_reg(self, clf, x_vars, ascending_order=True):

        self.df_train['group'] = 'training'
        self.df_valid['group'] = 'validation'

        df_combined = pd.concat([self.df_train, self.df_valid])

        Xp = df_combined[x_vars]
        Xp = Xp.astype(np.float32)
        pred_act = clf.predict(Xp)
        df_combined['p_score'] = pred_act

        df_combined.reset_index(inplace=True, drop=True)
        df_combined['p_rank'] = df_combined.groupby(['year'])['p_score'].rank(ascending=ascending_order)


        return df_combined

    @staticmethod
    def logit(x):
        return 1.0/(1.0 + np.exp(-x))

    def score_train_and_validation_class(self, clf, x_vars, ascending_order=True):

        self.df_train['group'] = 'training'
        self.df_valid['group'] = 'validation'

        df_combined = pd.concat([self.df_train, self.df_valid])

        Xp = df_combined[x_vars]
        Xp = Xp.astype(np.float32)
        pred_act = clf.predict_proba(Xp)
        pred_act1 = [x[1] for x in pred_act]
        df_combined['p_score'] = pred_act1
        df_combined['p_score'] = df_combined['p_score'].apply(self.logit)

        df_combined.reset_index(inplace=True, drop=True)
        df_combined['p_rank'] = df_combined.groupby(['year'])['p_score'].rank(ascending=ascending_order)

        return df_combined


    def score_pred_reg(self, clf, x_vars, ascending_order=True):
        Xp = self.df_score[x_vars]
        Xp = Xp.astype(np.float32)
        pred_act = clf.predict(Xp)
        self.df_score['p_score'] = pred_act


        self.df_score.reset_index(inplace=True, drop=True)
        self.df_score['p_rank'] = self.df_score.groupby(['year'])['p_score'].rank(ascending=ascending_order)

        return self.df_score


    def score_pred_class(self, clf, x_vars, ascending_order=True):
        Xp = self.df_score[x_vars]
        Xp = Xp.astype(np.float32)
        pred_act = clf.predict_proba(Xp)
        pred_act1 = [x[1] for x in pred_act]

        self.df_score['p_score'] = pred_act1
        self.df_score['p_score'] = self.df_score['p_score'].apply(self.logit)

        self.df_score.reset_index(inplace=True, drop=True)
        self.df_score['p_rank'] = self.df_score.groupby(['year'])['p_score'].rank(ascending=ascending_order)


        return self.df_score