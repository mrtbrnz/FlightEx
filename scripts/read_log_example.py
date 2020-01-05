#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, interp1d
import pprz_message_definitions as msg


class PPRZ_Data:
    """
    Data class from Paparazzi System.
    """
    def __init__(self, filename=None, ac_id=None, data_type=None):
        self.df_list = []
        self.filename = filename
        self.ac_id = ac_id
        self.df = None
        self.data_values = 0.
        self.data_type = data_type
        if self.data_type=='fault':
            self.read_msg1_bundle()
        elif self.data_type=='flight':
            self.read_msg1_bundle()
            self.read_msg2_bundle()
        self.find_min_max()
        self.df_All = self.combine_dataframes()
        
        
    def read_msg1_bundle(self):
        try:
            msg_name = 'attitude' ;columns=['time', 'phi','psi','theta'] ;drop_columns = ['time']
            self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
        except: print(' Attitude msg doesnt exist ')
        try:
            msg_name= 'mode'; columns=['time','mode','1','2','3','4','5']; drop_columns = ['time','1','2','3','4','5']
            self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
        except: print('Paparazzi Mode msg doesnt exist ')
        try:
            msg_name = 'imuaccel';columns=['time','Ax','Ay','Az']; drop_columns = ['time']
            self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
        except: print(' IMU Acceleration msg doesnt exist ')
        try:
            msg_name='gps';columns=['time','1','east','north','course','alt', 'vel', 'climb', '8','9','10','11'];drop_columns=['time','1','8','9','10','11']
            df = self.extract_message( msg_name, columns, drop_columns)
            df.alt = df.alt/1000.
            self.df_list.append(df)
        except: print(' GPS msg doesnt exist ')
        try:
            msg_name = 'imugyro';columns=['time','Gx','Gy','Gz']; drop_columns = ['time']
            self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
        except: print(' IMU Gyro msg doesnt exist ')

        
    def get_settings(self):
        ''' Special Message used for the fault injection settings
        2 multiplicative, and 2 additive, and only appears when we cahnge them
        so the time between has to be filled in...'''
        msg_name = 'settings'; columns=['time','m1','m2','add1','add2'];drop_columns=['time']
        df = self.extract_message( msg_name, columns, drop_columns)
        df.add1 = df.add1/9600. ; df.add2 = df.add2/9600.
        return df
       
    def extract_message(self, msg_name, columns, drop_columns):
        ''' Given msg names such as attitute, we will call msg.read_log_attitute'''
        exec('self.data_values = msg.read_log_{}(self.ac_id, self.filename)'.format(msg_name))
        df = pd.DataFrame(self.data_values, columns=columns)
        df.index = df.time
        df.drop(drop_columns, axis=1, inplace=True)
        return df
        
    def find_min_max(self):
        self.min_t = 1000.
        self.max_t = -1.
        for df in self.df_list:
            self.min_t = min(self.min_t, min(df.index))
            self.max_t = max(self.max_t, max(df.index))
        print('Min time :',self.min_t,'Maximum time :', self.max_t) # Minimum time can be deceiving... we may need to find a better way.

    def linearize_time(self, df, min_t=None, max_t=None, pad=10, period=0.01):
        if (min_t or max_t) == None:
            min_t = min(df.index)
            max_t = max(df.index)
        time = np.arange(int(min_t)+pad, int(max_t)-pad, period)
        out = pd.DataFrame()
        out['time'] = time
        for col in df.columns:
            func = interp1d(df.index , df[col]) # FIXME : If we want to use a different method other than linear interpolation.
            out[col] = func(time)
        out.index = out.time
        out.drop(['time'], axis=1, inplace=True)
        return out
    
    def combine_dataframes(self):
        frames = [self.linearize_time(df, self.min_t, self.max_t) for df in self.df_list]
        return pd.concat(frames, axis=1, ignore_index=False, sort=False)

    def combine_settings_dataframe(self):
        df_settings = self.get_settings() #FIXME : we may check if this has been already done before or not...
        return pd.concat(([self.df_All, df_settings]), axis=1, ignore_index=False, sort=False)

    def get_labelled_data(self):
        return self.combine_settings_dataframe().ffill()

def plot_all(data):
    import matplotlib
    import matplotlib.pyplot as plt
    # %config InlineBackend.figure_format = 'retina'
    import matplotlib as mpl
    mpl.style.use('default')
    import seaborn #plotting lib, but just adding makes the matplotlob plots better

    # fig=plt.figure(figsize=(19,7))
    # df_labelled.plot(y=['m1', 'alt'], figsize=(17,7));plt.show()
    data.plot(subplots=True, figsize=(17,25));plt.show()

def main():
    ac_id = '7'
    filename = '../Flight_Data/18_06_01__10_51_00_SD.data'
    data = PPRZ_Data(filename, ac_id, data_type='fault')
    plot_all(data.df_All)


if __name__ == "__main__":
    main()

