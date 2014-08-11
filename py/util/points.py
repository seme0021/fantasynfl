__author__ = 'msemeniuk'
import pandas as pd


class Points(classmethod):
    def __init__(self):
        pass

    @staticmethod
    def calc_passing_points(df):

        passing = {'TD': 6.0,
                   'YDS10': 1.0,
                   'YDS25': 1.0,
                   'INT': -2.0}

        res = {'points_td': 0,
               'points_yds10': 0,
               'points_yds25': 0,
               'points_int': 0,
               'points_total': 0}

        try:
            yds10 = round(df['YDS']/10) * passing['YDS10']
        except:
            yds10 = 0

        try:
            yds25 = round(df['YDS']/25) * passing['YDS25']
        except:
            yds25 = 0

        try:
            td = df['TD'] * passing['TD']
        except:
            td = 0

        try:
            inter = df['INT'] * passing['INT']
        except:
            inter = 0

        res['points_td'] = td
        res['points_yds10'] = yds10
        res['points_yds25'] = yds25
        res['points_int'] = inter
        res['points_total'] = td + yds10 + yds25 + inter

        return pd.Series(res)

    @staticmethod
    def calc_rec_points(df):

        passing = {'TD': 6.0,
                   'YDS10': 1.0,
                   'YDS25': 1.0}

        res = {'points_td': 0,
               'points_yds10': 0,
               'points_yds25': 0,
               'points_total': 0}

        try:
            yds10 = round(df['YDS']/10) * passing['YDS10']
        except:
            yds10 = 0

        try:
            yds25 = round(df['YDS']/25) * passing['YDS25']
        except:
            yds25 = 0

        try:
            td = df['TD'] * passing['TD']
        except:
            td = 0

        res['points_td'] = td
        res['points_yds10'] = yds10
        res['points_yds25'] = yds25
        res['points_total'] = td + yds10 + yds25

        return pd.Series(res)

