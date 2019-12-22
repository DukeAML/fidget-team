#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:03:26 2019

@author: jiarongchua92
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


raw_dataset = pd.read_csv("~/Fidget/users.csv", parse_dates = ['Date'])
raw_dataset_new = pd.read_csv("~/Fidget/users_new.csv", parse_dates = ['Date'])
refined_dataset = pd.DataFrame()
refined_dataset_new = pd.DataFrame()


def data_extraction(raw_df, refined_df):
    refined_df['Date'] = raw_df["Date"]
    refined_df['Time'] = raw_df["Time"]
    refined_df['Trace'] = raw_df["TraceTime"]
    refined_df['Quit'] = raw_df["QuitClicked"]
    refined_df['Start'] = raw_df["StartClicked"]
    refined_df['Setting'] = raw_df["SettingClicked"]
    
    # single values based variables but stored in a single element list
    refined_df['WristTiltGesture'] = raw_df["Wrist Tilt Gesture"]
    refined_df['HeartRateMonitor'] = raw_df["Heart Rate Monitor -Wakeup Secondary"]
    refined_df['StepCounter'] = raw_df["Step Counter"]
    refined_df['StepCounterSecondary'] = raw_df["Step Counter -Wakeup Secondary"]
    refined_df['StepDetector'] = raw_df["Step Detector"]
    refined_df['SignificantMotionDetector'] = raw_df["Significant Motion Detector"]
    refined_df['1000'] = raw_dataset["1000"]
    refined_df['1001'] = raw_dataset["1001"]
    refined_df['1002'] = raw_dataset["1002"]

    # vector based variables
    refined_df['RMD'] = raw_df["RMD"]
    refined_df['GameRotationVector'] = raw_df["Game Rotation Vector"]
    refined_df['LinearAcceleration'] = raw_df["Linear Acceleration"]
    refined_df['RotationVector'] = raw_df["Rotation Vector"]
    refined_df['CoarseMotionClassifier'] = raw_df["Coarse Motion Classifier"]
    refined_df['Pedometer'] = raw_df["Pedometer"]
    refined_df['LSM6DS3 Accelerometer'] = raw_df['LSM6DS3 Accelerometer']

    return(refined_df)

def parse_vectors(refined_dataset):
    refined_dataset["StepCounter"] = refined_dataset['StepCounter'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset["HeartRateMonitor"] = refined_dataset['HeartRateMonitor'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset["StepCounterSecondary"] = refined_dataset['StepCounterSecondary'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset["GameRotationVector"] = refined_dataset["GameRotationVector"].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
    refined_dataset['StepDetector'] = refined_dataset['StepDetector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset['RotationVector'] = refined_dataset['RotationVector'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
    refined_dataset['SignificantMotionDetector'] = refined_dataset['SignificantMotionDetector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset['LinearAcceleration'] = refined_dataset['LinearAcceleration'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
    refined_dataset['Time'] = pd.to_datetime(refined_dataset.Time, format="%H:%M:%S.%f").apply(lambda x: x.time())
    refined_dataset['LSM6DS3 Accelerometer'] = refined_dataset['LSM6DS3 Accelerometer'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))

    return(refined_dataset)
    

'''
November Data
'''

refined_dataset = data_extraction(raw_dataset, refined_dataset)
refined_dataset = refined_dataset.fillna(0)

refined_dataset = parse_vectors(refined_dataset)


refined_dataset_new = data_extraction(raw_dataset_new, refined_dataset_new)
refined_dataset_new = refined_dataset_new.fillna(0)

refined_dataset_new = parse_vectors(refined_dataset_new)



'''
11/07 Data
'''
subset_1107 = refined_dataset[refined_dataset['Date'] == '2019-11-07']
subset_1107_08 = subset_1107[subset_1107['Time'] < pd.to_datetime('08:00:00.000').time()]
#subset_1107_08['StepCounter'] = subset_1107_08['StepCounter'].replace(0, np.nan)
#subset_1107_08['StepCounter'].fillna(method = 'bfill', inplace = True)
#subset_1107_08['StepCounter'].fillna(0, inplace = True)

'''
11/15 Data - consider choosing another date
'''
subset_1115 = refined_dataset_new[refined_dataset_new['Date'] == '2019-11-15']
subset_1115 = subset_1115[subset_1115['Time'] > pd.to_datetime('06:00:00.000').time()]
#subset_1115['StepCounter'] = subset_1115['StepCounter'].replace(0, np.nan)
#subset_1115['StepCounter'].fillna(method = 'bfill', inplace = True)
#subset_1115['StepCounter'].fillna(0, inplace = True)

'''
11/09 Data
'''
subset_1109 = refined_dataset[refined_dataset['Date'] == '2019-11-09']
subset_1109 = subset_1109[subset_1109['Time'] > pd.to_datetime('06:00:00.000').time()]
subset_1109 = subset_1109[subset_1109['Time'] < pd.to_datetime('09:00:00.000').time()]
#subset_1109['StepCounter'] = subset_1109['StepCounter'].replace(0, np.nan)
#subset_1109['StepCounter'].fillna(method = 'ffill', inplace = True)
#subset_1109['StepCounter'].fillna(0, inplace = True)

vector_1115 = pd.DataFrame()
vector_1109 = pd.DataFrame()
vector_1107 = pd.DataFrame()

def create_linacc_subset(subset_df, vector_df):
    tmp = subset_df['LinearAcceleration'].apply(pd.Series)
    tmp = tmp.rename(columns = lambda x : 'linearacc_' + str(x))
    vector_df = pd.concat([vector_df, tmp])
    vector_df.dropna(inplace = True) # drop rows with no data collected
    vector_df['Time'] = subset_df['Time']
    return(vector_df)
    
vector_1115 = create_linacc_subset(subset_1115, vector_1115)
vector_1109 = create_linacc_subset(subset_1109, vector_1109)
vector_1107 = create_linacc_subset(subset_1107, vector_1107)

def create_accel_subset(subset_df, vector_df):
    tmp = subset_df['LSM6DS3 Accelerometer'].apply(pd.Series)
    tmp = tmp.rename(columns = lambda x : 'acc_' + str(x))
    vector_df = pd.concat([vector_df, tmp])
    vector_df['acc_0_sq'] = vector_df['acc_0'] ** 2
    vector_df['acc_1_sq'] = vector_df['acc_1'] ** 2
    vector_df['acc_2_sq'] = vector_df['acc_2'] ** 2
    vector_df['Magnitude'] = np.sqrt(vector_df[['acc_0_sq', 'acc_1_sq', 'acc_2_sq']].sum(axis = 1))
    vector_df.dropna(inplace = True)
    vector_df['ratio_x'] = vector_df['acc_0']/vector_df['Magnitude']
    vector_df['ratio_y'] = vector_df['acc_1']/vector_df['Magnitude']
    vector_df['angle_x'] = np.arccos(vector_df['ratio_x'])
    vector_df['angle_y'] = np.arccos(vector_df['ratio_y'])
    vector_df['Time'] = subset_df['Time']
    return(vector_df)


vector_1115 = create_accel_subset(subset_1115, vector_1115)
vector_1109 = create_accel_subset(subset_1109, vector_1109)
vector_1107 = create_accel_subset(subset_1107, vector_1107)


'''
rot_vector_1115 = pd.DataFrame()
RotationVector = subset_1115['RotationVector'].apply(pd.Series)
RotationVector = RotationVector.rename(columns = lambda x : 'vec_' + str(x))
rot_vector_1115  = pd.concat([rot_vector_1115 , RotationVector])
rot_vector_1115 .fillna(0, inplace = True)
rot_vector_1115['Time'] = subset_1115['Time']

rot_vector_1109 = pd.DataFrame()
RotationVector = subset_1109['RotationVector'].apply(pd.Series)
RotationVector = RotationVector.rename(columns = lambda x : 'vec_' + str(x))
rot_vector_1109  = pd.concat([rot_vector_1109 , RotationVector])
rot_vector_1109 .fillna(0, inplace = True)
rot_vector_1109['Time'] = subset_1109['Time']

rot_vector_1107 = pd.DataFrame()
RotationVector = subset_1107['RotationVector'].apply(pd.Series)
RotationVector = RotationVector.rename(columns = lambda x : 'vec_' + str(x))
rot_vector_1107  = pd.concat([rot_vector_1107 , RotationVector])
rot_vector_1107 .fillna(0, inplace = True)
rot_vector_1107['Time'] = subset_1107['Time']
'''


# calculate summary statistics
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,15))
'''
for i in range(0, 3):
    for j in range(0, 3):
        fig.add_subplot(axes[i,j])
'''
g1 = fig.add_subplot(axes[0,0])
g1.plot_date(subset_1107['Time'], subset_1107['StepCounter'])
g2 = fig.add_subplot(axes[0,1])
g2.plot_date(subset_1109['Time'], subset_1109['StepCounter'])
g3 = fig.add_subplot(axes[0,2])
g3.plot_date(subset_1115['Time'], subset_1115['StepCounter'])
g4 = fig.add_subplot(axes[1,0])
g4.plot_date(subset_1107['Time'], subset_1107['HeartRateMonitor'])
g5 = fig.add_subplot(axes[1,1])
g5.plot_date(subset_1109['Time'], subset_1109['HeartRateMonitor'])
g6 = fig.add_subplot(axes[1,2])
g6.plot_date(subset_1115['Time'], subset_1115['HeartRateMonitor'])
g7 = fig.add_subplot(axes[2,0])
g7.plot_date(vector_1107['Time'], vector_1107['Magnitude'])
g8 = fig.add_subplot(axes[2,1])
g8.plot_date(vector_1109['Time'], vector_1109['Magnitude'])
g9 = fig.add_subplot(axes[2,2])
g9.plot_date(vector_1115['Time'], vector_1115['Magnitude'])
g10 = fig.add_subplot(axes[3,0])
g10.plot_date(vector_1107['Time'], vector_1107['angle_x'])
g11 = fig.add_subplot(axes[3,1])
g11.plot_date(vector_1109['Time'], vector_1109['angle_x'])
g12 = fig.add_subplot(axes[3,2])
g12.plot_date(vector_1115['Time'], vector_1115['angle_x'])


g1.title.set_text('11/07')
g2.title.set_text('11/09')
g3.title.set_text('11/15')
g1.set(ylabel = 'StepCounter')
g4.set(ylabel = 'HeartRate')
g7.set(ylabel = 'AccelerationMagnitude(m/s^2)')
g10.set(ylabel = 'Tilt Angle(rad)')

plt.savefig('watchdata_x.png')


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,15))
'''
for i in range(0, 3):
    for j in range(0, 3):
        fig.add_subplot(axes[i,j])
'''
g1 = fig.add_subplot(axes[0,0])
g1.plot_date(subset_1107['Time'], subset_1107['StepCounter'])
g2 = fig.add_subplot(axes[0,1])
g2.plot_date(subset_1109['Time'], subset_1109['StepCounter'])
g3 = fig.add_subplot(axes[0,2])
g3.plot_date(subset_1115['Time'], subset_1115['StepCounter'])
g4 = fig.add_subplot(axes[1,0])
g4.plot_date(subset_1107['Time'], subset_1107['HeartRateMonitor'])
g5 = fig.add_subplot(axes[1,1])
g5.plot_date(subset_1109['Time'], subset_1109['HeartRateMonitor'])
g6 = fig.add_subplot(axes[1,2])
g6.plot_date(subset_1115['Time'], subset_1115['HeartRateMonitor'])
g7 = fig.add_subplot(axes[2,0])
g7.plot_date(vector_1107['Time'], vector_1107['Magnitude'])
g8 = fig.add_subplot(axes[2,1])
g8.plot_date(vector_1109['Time'], vector_1109['Magnitude'])
g9 = fig.add_subplot(axes[2,2])
g9.plot_date(vector_1115['Time'], vector_1115['Magnitude'])
g10 = fig.add_subplot(axes[3,0])
g10.plot_date(vector_1107['Time'], vector_1107['angle_y'])
g11 = fig.add_subplot(axes[3,1])
g11.plot_date(vector_1109['Time'], vector_1109['angle_y'])
g12 = fig.add_subplot(axes[3,2])
g12.plot_date(vector_1115['Time'], vector_1115['angle_y'])

g1.title.set_text('11/07')
g2.title.set_text('11/09')
g3.title.set_text('11/15')
g1.set(ylabel = 'StepCounter')
g4.set(ylabel = 'HeartRate')
g7.set(ylabel = 'AccelerationMagnitude(m/s^2)')
g10.set(ylabel = 'Tilt Angle(rad)')

plt.savefig('watchdata_y.png')

