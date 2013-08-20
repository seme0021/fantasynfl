import numpy as np
import pandas as pd

class CostFunction:
    def __init__(self, df, y, yhat, group_by, min_group_by_n, report_by=None, std_ol=None, std_oh=None):
        """
        CostFunction class contains all relevant cost functions.
        df: Pandas DataFrame that has actual and predicted values
        y: Actual/Dependent variable
        yhat: Predicted variable
        group_by: If computing volume-weighted metrics, specify the group you'd like to subset by.
        min_group_by: if computing volume-weighted metrics, specify the minimum number of observations per group.
        std_ol: If you'd like to remove outliers from your predictions, specify the min standard deviation.
        std_oh: Max standard deviation for outlier removal
        """

        self.df = df
        self.y = y
        self.yhat = yhat
        self.group_by = group_by
        self.min_group_by_n = min_group_by_n
        self.report_by = report_by
        self.std_ol = std_ol
        self.std_oh = std_oh

    def cf_rsquared(self):
        """
        Computes the R-Squared of the actual vs predicted values.
        """

        df = self.df

        y_mean = self.df[self.y].mean()
        n = len(df)
        df['ss_tot'] = (df[self.y] - y_mean) ** 2
        ss_tot = df['ss_tot'].sum()

        df['ss_err'] = (df[self.y] - df[self.yhat]) ** 2
        ss_err = df['ss_err'].sum()

        ss_reg = ((df[self.yhat] - y_mean)**2).sum()

        #rsq = 1 - (ss_err / ss_tot)
        rsq = ss_reg / ss_tot

        return rsq


    def cf_mpe(self):
        """
        Computes the Mean Percent Error (MPE)
        """
        df = self.df
        df['residual'] = (df[self.yhat] - df[self.y]) / df[self.y]
        if self.std_ol is not None and self.std_oh is not None:
            #Remove Outliers
            error_mean = df['residual'].describe()['mean']
            error_std = df['residual'].describe()['std']
            df = self.df[(df['residual'] < error_mean + error_std * self.std_oh) &
                         (df['residual'] > error_mean - error_std * self.std_ol)]
        mpe = df['residual'].mean()

        return mpe

    def cf_rmse(self):
        """
        Computes Root-Mean-Squared-Error (RMSE)
        df: pandas data frame
        y: dependent variable
        yhat: predicted variable
        std_ol: Outlier (max) of error
        std_oh: Outlier (min) of error
        """
        #Compute the error
        df = self.df
        if self.std_ol is not None and self.std_oh is not None:
            df['error'] = df[self.y] - df[self.yhat]
            mean_error = df['error'].describe()['mean']
            std_error = df['error'].describe()['std']
            df = df[(df['error'] < mean_error + std_error * self.std_oh) &
                    (df['error'] > mean_error - std_error * self.std_ol)]
        df['sq_error'] = (df[self.y] - df[self.yhat]) ** 2
        rmse = np.sqrt(df['sq_error'].describe()['mean'])

        return rmse

    def cf_mape(self):
        """
        Compute Mean Absolute Percent Error.
        """
        df = self.df
        df['abs_residual'] = np.abs((df[self.y] - df[self.yhat]) / df[self.y])
        if self.std_oh is not None:
            #Remove Outliers
            error_mean = df['abs_residual'].describe()['mean']
            error_std = df['abs_residual'].describe()['std']
            df = df[df['abs_residual'] < error_mean + error_std * self.std_oh]
        mape = df['abs_residual'].mean()
        return mape

    def cf_vw_mape(self):
        """
        Computes volume weighted MAPE.
        """
        df = self.df
        df['mape'] = np.abs((df[self.y] - df[self.yhat]) / df[self.y])
        if self.std_oh is not None:
            error_mean = df['mape'].describe()['mean']
            error_std = df['mape'].describe()['std']
            df = df[df['mape'] < error_mean + error_std * self.std_oh]
        df_grouped = df[[self.group_by, self.y, self.yhat, 'mape']].groupby(self.group_by).agg([np.mean, np.std, np.size])
        df_grouped = df_grouped[df_grouped['mape']['size'] >= self.min_group_by_n]
        df_grouped['mean_sum'] = df_grouped['mape']['size'] * df_grouped['mape']['mean']

        vw_mape = df_grouped.sum()['mean_sum'] / df_grouped.sum()['mape']['size']

        return vw_mape

    def cf_vw_mape_by_group(self):
        """
        Computes volume weighted MAPE and produces a report given a 'report_by' group.
        """
        self.df['mape'] = np.abs((self.df[self.y] - self.df[self.yhat]) / self.df[self.y])
        if self.std_oh is not None:
            error_mean = self.df['mape'].describe()['mean']
            error_std = self.df['mape'].describe()['std']
            df = self.df[self.df['mape'] < error_mean + error_std * self.std_oh]
        df_grouped = df[[self.report_by, self.group_by, self.y, self.yhat, 'mape']].groupby([self.report_by, self.group_by]).agg([np.mean, np.std, np.size])
        df_grouped = df_grouped[df_grouped['mape']['size'] >= self.min_group_by_n]
        df_grouped.reset_index(inplace=True, drop=False)
        df_grouped['mean_sum'] = df_grouped['mape']['size'] * df_grouped['mape']['mean']
        df_grouped['mape_size'] = df_grouped['mape']['size']

        #This is bad, but only way i could figure out how to make group by work again...
        dft = pd.DataFrame(df_grouped['year'])
        dft['bin_id'] = df_grouped['bin_id']
        dft['mean_sum'] = df_grouped['mean_sum']
        dft['mape_size'] = df_grouped['mape_size']

        grp = dft[['year', 'mean_sum', 'mape_size']].groupby(['year']).agg(np.sum)
        grp['vw_mape'] = grp['mean_sum']/grp['mape_size']
        return grp['vw_mape']

    def cf_mpsd(self):
        """
        Computes Mean Percent Standard Deviation.
        """
        df = self.df
        df['mpe'] = (df[self.yhat] - df[self.y]) / df[self.y]
        if self.std_ol is not None and self.std_oh is not None:
            #Remove Outliers
            error_mean = df['mpe'].describe()['mean']
            error_std = df['mpe'].describe()['std']
            df = df[(df['mpe'] < error_mean + error_std * self.std_oh) &
                    (df['mpe'] > error_mean - error_std * self.std_ol)]
        mpsd = df['mpe'].describe()['std']

        return mpsd

    def all_glm(self):
        """
        Returns all cost functions relevant to a GLM model.
        """
        rsq = self.cf_rsquared()
        mpe = self.cf_mpe()
        mape = self.cf_mape()
        vw_mape = self.cf_vw_mape()
        mpsd = self.cf_mpsd()
        rmse = self.cf_rmse()

        cost_functions = {'r_squared': rsq,
                          'mpe': mpe,
                          'mape': mape,
                          'vw_mape': vw_mape,
                          'mpsd': mpsd,
                          'rmse': rmse
        }
        return cost_functions