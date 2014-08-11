__author__ = 'msemeniuk'
import numpy as np
import pandas as pd

class Methods(classmethod):
    def __init__(self):
        self.rank='RK'

    @staticmethod
    def topx(x, n):
        if x <= n:
            return 1
        return 0

    @staticmethod
    def sqrt_rank(x):
        return np.sqrt(x)

    @staticmethod
    def is_pos(player, position_string):
        """
        Checks if the position string (i.e. QB) is in the player string
        :param player: str
        :param position_string: str
        :return: int
        """
        if position_string in player:
            return 1
        return 0

    @staticmethod
    def is_quarterback(x):
        """
        1/0 if quarterback
        :param x: str
        :return: int
        """
        if 'qb' in str(x).lower():
            return 1
        return 0


    @staticmethod
    def is_receiver(x):
        """
        1/0 if quarterback
        :param x: str
        :return: int
        """
        if 'wr' in str(x).lower():
            return 1
        return 0

    @staticmethod
    def is_runner(x):
        """
        1/0 if quarterback
        :param x: str
        :return: int
        """
        if 'rb' in str(x).lower():
            return 1
        return 0


    @staticmethod
    def convert_to_float(x):
        """
        Convert string to float
        :param x: str
        :return: float
        """
        try:
            v = float(x.replace(',', ''))
        except:
            v = x


        return v


    @staticmethod
    def convert_to_int(x):
        try:
            return int(str(x).replace(',', ''))
        except ValueError:
            return np.nan
        except TypeError:
            return np.nan

    @staticmethod
    def has_rank(x, max_val):
        """
        :param x:
        :param max_val: int
        :return:
        """
        try:
            if 0 <= int(x) <= max_val:
                return 1
        except:
            return 0
        return 0

    @staticmethod
    def games_played(df):
        try:
            gp = int(df['YDS'] / df['YDS/G'])
            return gp
        except:
            return 0



    @staticmethod
    def games_played_rb(df):
        try:
            gp = int(df['YDS'] / df['YDS/G'])
            return gp
        except:
            return 0

    @staticmethod
    def last_rank_detla(df):
        """
        Assumes that the input variables have been validated
        :param df:
        :return:
        """
        return ( df['RK'] - df['previous_rank1_clean']) / df['previous_rank1_clean']



    @staticmethod
    def clean_up_prior_ranks(df):
        res = {'previous_rank1_clean': df['previous_rank1'],
               'previous_rank2_clean': df['previous_rank2'],
               'previous_rank3_clean': df['previous_rank3'],
               'previous_rank4_clean': df['previous_rank4'],
               'previous_rank5_clean': df['previous_rank5'],
               'avg_career_rank': np.NaN,
               'has_previous_rank1': 0,
               'has_previous_rank2': 0,
               'has_previous_rank3': 0,
               'has_previous_rank4': 0,
               'has_previous_rank5': 0
        }

        #Set Previous Rank 1 Variables
        if pd.isnull(res['previous_rank1_clean']):
            res['previous_rank1_clean'] = df['RK']
            res['has_previous_rank1'] = 0
        elif pd.isnull(res['previous_rank1_clean']) == False:
            res['has_previous_rank1'] = 1

        #Set Previous Rank 2 Variables
        if pd.isnull(res['previous_rank2_clean']):
            res['previous_rank2_clean'] = res['previous_rank1_clean']
            res['has_previous_rank2'] = 0
        elif pd.isnull(res['previous_rank2_clean']) == False:
            res['has_previous_rank2'] = 1

        #Set Previous Rank 3 Variables
        if pd.isnull(res['previous_rank3_clean']):
            res['previous_rank3_clean'] = res['previous_rank2_clean']
            res['has_previous_rank3'] = 0
        elif pd.isnull(res['previous_rank3_clean']) == False:
            res['has_previous_rank3'] = 1

        #Set Previous Rank 4 Variables
        if pd.isnull(res['previous_rank4_clean']):
            res['previous_rank4_clean'] = res['previous_rank3_clean']
            res['has_previous_rank4'] = 0
        elif pd.isnull(res['previous_rank4_clean']) == False:
            res['has_previous_rank4'] = 1

        #Set Previous Rank 4 Variables
        if pd.isnull(res['previous_rank5_clean']):
            res['previous_rank5_clean'] = res['previous_rank4_clean']
            res['has_previous_rank5'] = 0
        elif pd.isnull(res['previous_rank5_clean']) == False:
            res['has_previous_rank5'] = 1

        res['avg_career_rank'] = np.mean([df['RK'], res['previous_rank1_clean'], res['previous_rank2_clean'],
                                          res['previous_rank3_clean'], res['previous_rank4_clean'],
                                          res['previous_rank5_clean']])

        return pd.Series(res)

