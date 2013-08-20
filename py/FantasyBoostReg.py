import pandas as pd
import numpy as np
import pylab as pl
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('/Users/msemeniuk/git/YHandler/')

from cost_functions import CostFunction
pd.set_option('display.line_width', 300)


class FantasyBoostReg:
    def __init__(self, path, file_prefix):
        self.path = path
        self.file_prefix = file_prefix

    @staticmethod
    def _is_qb(x):
        if 'QB' in x:
            return 1
        return 0

    @staticmethod
    def _to_float(x):
        try:
            v = float(x.replace(',', ''))
        except:
            v = x
        return v

    @staticmethod
    def _to_int(x):
        try:
            v = int(x.replace(',', ''))
        except:
            v = x
        return v

    @staticmethod
    def _to_int(x):
        try:
            return int(str(x).replace(',', ''))
        except ValueError:
            return np.nan

    @staticmethod
    def _has_rank(x):
        try:
            if 0 <= int(x) <= 40:
                return 1
        except:
            return 0
        return 0

    @staticmethod
    def _games_played(df):
        try:
            gp = int(df['YDS'] / df['YDS/G'])
            return gp
        except:
            return 0

    def build_research_dataset(self, min_year, max_year):
        yrs = range(min_year, max_year + 1)
        yrs.reverse()
        first = True
        cols = ['RK', 'PLAYER', 'TEAM', 'COMP', 'ATT', 'PCT', 'YDS',
                'YDS/A', 'LONG', 'TD', 'INT', 'SACK', 'RATE', 'YDS/G']

        for y in yrs:
            if first:
                df = pd.read_csv(self.path + '/' + self.file_prefix + str(y) + '.txt',
                                 sep='\t',
                                 index_col=None,
                                 names=cols)
                df['year'] = int(y)
            elif not first:
                dft = pd.read_csv(self.path + '/' + self.file_prefix + str(y) + '.txt',
                                  sep='\t',
                                  index_col=None,
                                  names=cols)
                dft['year'] = int(y)
                df = df.append(dft)
            first = False

        to_int_vars = ['RK', 'COMP', 'ATT', 'YDS', 'LONG', 'TD', 'INT', 'SACK']
        to_flt_vars = ['PCT', 'YDS/A', 'YDS/G', 'RATE']

        #Convert fields to either int or float
        for i in to_int_vars:
            df[i] = df[i].apply(self._to_int)

        for i in to_flt_vars:
            df[i] = df[i].apply(self._to_float)

        #Feature extraction
        df['is_qb'] = df['PLAYER'].apply(self._is_qb)
        df['has_rank'] = df['RK'].apply(self._has_rank)
        df['games_played'] = df.apply(self._games_played, axis=1)
        df = df[df['has_rank'] == 1]

        df.reset_index(inplace=True, drop=False)
        df.set_index('year', inplace=True, drop=False)

        #Feature extraction continued
        df = df.join(df.groupby('PLAYER')['year'].min(), on='PLAYER', rsuffix='_min')
        #Number of years played (for given season)
        df['n_years'] = df['year'] - df['year_min'] + 1
        #Number of touchdown's per game
        df['td_per_game'] = df['TD'] / df['games_played']
        #Number of Completions per game
        df['comp_per_game'] = df['COMP'] / df['games_played']
        #Number of interceptions per game
        df['int_per_game'] = df['INT'] / df['games_played']
        #Number of sacks per game
        df['sack_per_game'] = df['SACK'] / df['games_played']
        #Log total yards in season
        df['log_yards'] = np.log(df['YDS'])
        #Ratio Attempts/Completions
        df['r_att_comp'] = df['ATT'] / df['COMP']
        #Number of yards per game
        df['r_yards_game'] = df['YDS'] / df['games_played']
        #Number of attempts per game
        df['r_att_game'] = df['ATT'] / df['games_played']
        #Percent completions/100
        df['percent_comp'] = df['PCT'] / 100

        return df

    def build_dependent_var(self, df):
        #Count number of players
        dfg = df[['PLAYER', 'RK']].groupby('PLAYER').agg(np.size)
        dfg = dfg[dfg['RK'] >= 2]
        df.set_index('PLAYER', inplace=True, drop=False)

        dfg.rename(columns={'RK': 'n_years_tot'}, inplace=True)
        df = df.join(dfg)

        df.reset_index(inplace=True, drop=True)

        df['lst_RK'] = df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift())

        return df


    def build_train_valid_score(self, df, score_year, pct_train):
        df.set_index('year', inplace=True, drop=False)
        #Build Prediction Data
        df_pred = df.ix[2012]
        df.set_index('PLAYER', inplace=True, drop=False)
        df_pred.set_index('PLAYER', inplace=True, drop=False)

        #Define Train and Validation Datasets
        ix = shuffle(df.index)
        n_train = int(len(ix) * pct_train / 100)
        ix_t = ix[:n_train]
        ix_v = ix[n_train:]

        df_train = df.ix[ix_t]
        df_valid = df.ix[ix_v]

        df_train.set_index('PLAYER', inplace=True, drop=False)
        df_valid.set_index('PLAYER', inplace=True, drop=False)

        dsn_dict = {'train': df_train,
                    'valid': df_valid,
                    'score': df_pred
                    }
        return dsn_dict

    def train_model(self, df_train, params, x_vars, y_var):
        #Clean up the data a bit (remove records with null values)
        df_train = df_train[x_vars + y_var + ['PLAYER']].dropna()

        X = df_train[x_vars]
        y = df_train[y_var]
        X = X.astype(np.float32)

        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(X, y)
        return clf

    def score_valid(self, df_valid, clf, x_vars, y_var):
        df_valid = df_valid[x_vars + y_var + ['PLAYER']].dropna()
        Xv = df_valid[x_vars]
        Xv = Xv.astype(np.float32)
        df_valid['p_rank'] = clf.predict(Xv)
        return df_valid

    def feature_importance(self, clf, x_vars):
        feature_importance = clf.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        for i in sorted_idx:
            print np.array(x_vars)[i] + ' '* (20 - len(np.array(x_vars)[i])) + str(feature_importance[i])

    def score_pred(self, clf, df_pred, x_vars):
        Xp = df_pred[x_vars]
        Xp = Xp.astype(np.float32)
        pred_act = clf.predict(Xp)
        df_pred['p_rank'] = pred_act
        min_score = df_pred['p_rank'].min()
        df_pred['p_rank'] = df_pred['p_rank'] - (min_score - 1)

        result = df_pred.sort(['p_rank'], ascending=[1])
        result[x_vars + ['p_rank', 'TEAM']]

        return result

    def main(self):
        df_research = self.build_research_dataset(2002, 2012)
        df_research = self.build_dependent_var(df_research)
        df_tfs = self.build_train_valid_score(df_research, 2012, 70)

        #Initialize training and validation datasets
        x_vars = ['RK', 'comp_per_game', 'r_att_game', 'percent_comp', 'log_yards', 'YDS/A', 'LONG', 'td_per_game',
                  'int_per_game', 'sack_per_game', 'r_att_comp', 'r_yards_game', 'games_played', 'n_years', 'RATE']

        y_var = ['lst_RK']
        #y_var = ['log_lst_rk']

        params = {'n_estimators': 400,
                  'max_depth': 3,
                  'min_samples_split': 2,
                  'learning_rate': 0.04,
                  'loss': 'ls',
                  'verbose': 1}
        clf = self.train_model(df_tfs['train'], params=params, x_vars=x_vars, y_var=y_var)

        df_valid = df_tfs['valid']
        df_valid_scored = self.score_valid(df_valid, clf, x_vars, y_var)

        #cf = CostFunction(df_valid_scored, 'log_lst_rk', 'p_rank', 'PLAYER', 1)
        cf = CostFunction(df_valid_scored, 'lst_RK', 'p_rank', 'PLAYER', 1)

        df_pred_scored = self.score_pred(clf, df_tfs['score'], x_vars)
        df_pred_scored[['RK', 'p_rank', 'TEAM']]
        df_pred_scored['vw_mape'] = cf.cf_mape()
        df_pred_scored['r_sqrd'] = cf.cf_rsquared()

        results = df_pred_scored[['RK', 'p_rank', 'TEAM', 'vw_mape', 'r_sqrd']]
        self.feature_importance(clf, x_vars)

        return results

if __name__ == '__main__':
    first = True
    for i in range(1, 500):
        nfl = FantasyBoostReg('/Users/msemeniuk/nfl', 'passing_')
        if first:
            df_fnl = nfl.main()
        if not first:
            df_fnl = df_fnl.append(nfl.main())
        first = False
    df_fnl.reset_index(drop=False, inplace=True)
    r = df_fnl.groupby('PLAYER').agg(np.mean)
    r = r.sort(['p_rank'], ascending=[1])
    print r