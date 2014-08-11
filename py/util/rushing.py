__author__ = 'msemeniuk'
import pandas as pd
import numpy as np
from points import Points
from methods import Methods

class Rushing(classmethod):
    def __init__(self, min_year, max_year):
        self.min_year=min_year
        self.max_year=max_year
        self.max_rank=60
        self.topn = 10
        self.path='../data'
        self.file_prefix='rushing_'
        self.df = pd.DataFrame()
        self.to_int_vars = ['Year', 'RK', 'ATT', 'YDS', 'LONG', '20+', 'TD', 'FUM', '1D']
        self.to_flt_vars = ['YDS/G', 'YDS/A']
        self.funcs=Methods()
        self.points=Points()
        self.feature_vector=[u'RK', u'is_runner', u'games_played', u'n_years', u'td_per_game', u'att_per_game',
                             u'r_yards_game', u'r_20p_game', u'r_fum_game', u'r_yds_att', u'log_long',
                             u'r_1d_game', u'avg_career_rank', u'previous_rank1_clean',
                             u'previous_rank2_clean', u'previous_rank3_clean', u'previous_rank4_clean',
                             u'previous_rank5_clean', u'lst_rank_delta', u'r_td_rec', u'log_yards']
        self.dep_var=[u'y_future_rank', u'y_top5', 'y_sqrt_future_rank', 'y_future_points', 'y_sqr_future_points']


    def compile_data(self):
        """
        :return: void
        """

        path = '../data'
        file_prefix = 'rushing_'
        yrs = range(self.min_year, self.max_year + 1)
        yrs.reverse()
        first = True

        cols = ['Year', 'RK', 'PLAYER', 'TEAM', 'ATT', 'YDS', 'YDS/A', 'LONG',
                '20+','TD', 'YDS/G', 'FUM', '1D']

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

    def get_receiving_data(self):
        """
        :return: pd.DataFrame
        """
        return self.df

    @staticmethod
    def _clean_value(x):

        if pd.isnull(x):
            return 0

        if x>1:
            return 1
        if x<0:
            return 0

        return x


    def construct_feature_vector(self):
        to_build = [  u'avg_career_rank', u'previous_rank1_clean',
                             u'previous_rank2_clean', u'previous_rank3_clean', u'previous_rank4_clean',
                             u'previous_rank5_clean', u'lst_rank_delta', u'r_td_rec', u'log_yards']

        #Feature extraction
        self.df['is_runner'] = self.df['PLAYER'].apply(self.funcs.is_runner)
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
        self.df['att_per_game'] = self.df['ATT'] / self.df['games_played']
        #Number of Yards per game
        self.df['r_yards_game'] = self.df['YDS'] / self.df['games_played']

        self.df['r_20p_game'] = self.df['20+'] / self.df['games_played']
        self.df['r_fum_game'] = self.df['FUM'] / self.df['games_played']

        self.df['r_1d_game'] = self.df['1D'] / self.df['games_played']

        self.df['r_yds_att'] = self.df['YDS'] / self.df['ATT']
        #self.df['r_yds_att'] = self.df['r_yds_att'].apply(self._clean_value)

        self.df['r_td_rec'] = self.df['TD'] / self.df['ATT']
        self.df['r_td_rec'] = self.df['r_td_rec'].apply(self._clean_value)


        #Log total yards in season
        self.df['log_yards'] = np.log(self.df['YDS'])
        self.df['log_long'] = np.log(self.df['LONG'])


        self.df.reset_index(inplace=True, drop=True)

        self.df['previous_rank1'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-1))
        self.df['previous_rank2'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-2))
        self.df['previous_rank3'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-3))
        self.df['previous_rank4'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-4))
        self.df['previous_rank5'] = self.df.groupby('PLAYER')['RK'].apply(lambda grp: grp.shift(periods=-5))


        self.df = self.df.join(self.df.apply(self.funcs.clean_up_prior_ranks, axis=1))
        self.df = self.df.join(self.df.apply(self.points.calc_rec_points, axis=1))

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