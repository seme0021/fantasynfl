__author__ = 'msemeniuk'
import pandas as pd
import numpy as np
from points import Points
from methods import Methods

class Passing(classmethod):
    def __init__(self, min_year, max_year):
        self.min_year=min_year
        self.max_year=max_year
        self.max_rank=60
        self.topn = 5
        self.path='../data'
        self.file_prefix='passing_'
        self.df = pd.DataFrame()
        self.to_int_vars = ['RK', 'COMP', 'ATT', 'YDS', 'LONG', 'TD', 'INT', 'SACK']
        self.to_flt_vars = ['YDS/G', 'AVG', 'RATE', 'PCT']
        self.funcs=Methods()
        self.points=Points()
        self.feature_vector=[u'RK', u'is_qb', u'games_played', u'n_years', u'td_per_game', u'comp_per_game',
                             u'int_per_game', u'sack_per_game', u'log_yards', u'r_att_comp', u'r_yards_game',
                             u'r_att_game', u'percent_comp', u'avg_career_rank', u'previous_rank1_clean',
                             u'previous_rank2_clean', u'previous_rank3_clean', u'previous_rank4_clean',
                             u'previous_rank5_clean', u'lst_rank_delta']
        self.dep_var=[u'y_future_rank', u'y_top5', 'y_sqrt_future_rank', 'y_future_points', 'y_sqr_future_points']


    def compile_data(self):
        """
        :return: void
        """

        path = '../data'
        file_prefix = 'passing_'
        yrs = range(self.min_year, self.max_year + 1)
        yrs.reverse()
        first = True

        cols = ['RK', 'PLAYER', 'TEAM', 'COMP', 'ATT', 'PCT', 'YDS', 'AVG',
                'LONG', 'TD', 'INT', 'SACK', 'RATE', 'YDS/G']

        #Build the research data set by importing individual
        #data files
        for y in yrs:
            if first:
                self.df = pd.read_csv(self.path + '/' + self.file_prefix + str(y) + '.txt',
                                 sep='\t',
                                 index_col=None,
                                 names=cols)
                self.df['year'] = int(y)
            elif not first:
                dft = pd.read_csv(path + '/' + file_prefix + str(y) + '.txt',
                                  sep='\t',
                                  index_col=None,
                                  names=cols)
                dft['year'] = int(y)
                self.df = self.df.append(dft)
            first = False

        #Convert fields to either int
        for i in self.to_int_vars:
            self.df[i] = self.df[i].apply(self.funcs.convert_to_int)

        #Convert fields to float
        for i in self.to_flt_vars:
            self.df[i] = self.df[i].apply(self.funcs.convert_to_float)

    def get_passing_data(self):
        """
        :return: pd.DataFrame
        """
        return self.df

    def construct_feature_vector(self):
         #Feature extraction
        self.df['is_qb'] = self.df['PLAYER'].apply(self.funcs.is_quarterback)
        self.df['has_rank'] = self.df['RK'].apply(self.funcs.has_rank, args=[self.max_rank])
        self.df['games_played'] = self.df.apply(self.funcs.games_played, axis=1)

        #Keep only those players that have a rank in the data
        self.df = self.df[self.df['has_rank'] == 1]

        self.df.reset_index(inplace=True, drop=False)
        self.df.set_index('year', inplace=True, drop=False)

        #Feature extraction continued
        self.df = self.df.join(self.df.groupby('PLAYER')['year'].min(), on='PLAYER', rsuffix='_min')
        #Number of years played (for given season)
        self.df['n_years'] = self.df['year'] - self.df['year_min'] + 1
        #Number of touchdown's per game
        self.df['td_per_game'] = self.df['TD'] / self.df['games_played']
        #Number of Completions per game
        self.df['comp_per_game'] = self.df['COMP'] / self.df['games_played']
        #Number of interceptions per game
        self.df['int_per_game'] = self.df['INT'] / self.df['games_played']
        #Number of sacks per game
        self.df['sack_per_game'] = self.df['SACK'] / self.df['games_played']
        #Log total yards in season
        self.df['log_yards'] = np.log(self.df['YDS'])
        #Ratio Attempts/Completions
        self.df['r_att_comp'] = self.df['ATT'] / self.df['COMP']
        #Number of yards per game
        self.df['r_yards_game'] = self.df['YDS'] / self.df['games_played']
        #Number of attempts per game
        self.df['r_att_game'] = self.df['ATT'] / self.df['games_played']
        #Percent completions/100
        self.df['percent_comp'] = self.df['PCT'] / 100

        self.df.reset_index(inplace=True, drop=True)

        self.df['previous_rank1'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-1))
        self.df['previous_rank2'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-2))
        self.df['previous_rank3'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-3))
        self.df['previous_rank4'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-4))
        self.df['previous_rank5'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-5))


        self.df = self.df.join(self.df.apply(self.funcs.clean_up_prior_ranks, axis=1))
        self.df = self.df.join(self.df.apply(self.points.calc_passing_points, axis=1))

        self.df['lst_rank_delta'] = self.df.apply(self.funcs.last_rank_detla, axis=1)
        #Compute dependent variable (set this current year's rank to last yaer's data
        self.df['y_future_rank'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift())
        self.df['y_future_points'] = self.df.groupby('PLAYER')['points_total'].apply(lambda grp: grp.shift())
        self.df['y_top5'] = self.df['y_future_rank'].apply(self.funcs.topx, args=[self.topn])
        self.df['y_sqrt_future_rank'] = self.df['y_future_rank'].apply(self.funcs.sqrt_rank)
        self.df['y_sqr_future_points'] = self.df['y_future_points'].apply(self.funcs.sqrt_rank)

        #Only keep data after 2005
        self.df = self.df[self.df.year >= 2005]

        self.df.__delitem__('previous_rank1')
        self.df.__delitem__('previous_rank2')
        self.df.__delitem__('previous_rank3')
        self.df.__delitem__('previous_rank4')
        self.df.__delitem__('previous_rank5')