#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:03:26 2019

@author: jiarongchua92
"""

import pandas as pd
raw_dataset = pd.read_csv("/Users/jiarongchua92/Downloads/users.csv", parse_dates = ['Date'])
raw_dataset_new = pd.read_csv("/Users/jiarongchua92/Downloads/users_new.csv", parse_dates = ['Date'])
raw_dataset_july = pd.read_csv("/Users/jiarongchua92/Downloads/users_july.csv", parse_dates = ['Date'])
refined_dataset = pd.DataFrame()
refined_dataset_new = pd.DataFrame()
refined_dataset_july = pd.DataFrame()

'''
Early November Data
'''

# single values based variables
refined_dataset['Date'] = raw_dataset["Date"]
refined_dataset['Time'] = raw_dataset["Time"]
refined_dataset['Trace'] = raw_dataset["TraceTime"]
refined_dataset['Quit'] = raw_dataset["QuitClicked"]
refined_dataset['Start'] = raw_dataset["StartClicked"]
refined_dataset['Setting'] = raw_dataset["SettingClicked"]
# single values based variables but stored in a single element list
refined_dataset['WristTiltGesture'] = raw_dataset["Wrist Tilt Gesture"]
refined_dataset['HeartRateMonitor'] = raw_dataset["Heart Rate Monitor -Wakeup Secondary"]
refined_dataset['StepCounter'] = raw_dataset["Step Counter"]
refined_dataset['StepCounterSecondary'] = raw_dataset["Step Counter -Wakeup Secondary"]
refined_dataset['StepDetector'] = raw_dataset["Step Detector"]
#refined_dataset['StepDetectorSecondary'] = raw_dataset["Step Detector -Wakeup Secondary"]
refined_dataset['SignificantMotionDetector'] = raw_dataset["Significant Motion Detector"]
refined_dataset['1000'] = raw_dataset["1000"]
refined_dataset['1001'] = raw_dataset["1001"]
refined_dataset['1002'] = raw_dataset["1002"]


# vector based variables
refined_dataset['RMD'] = raw_dataset["RMD"]
refined_dataset['GameRotationVector'] = raw_dataset["Game Rotation Vector"]
#refined_dataset['GameRotationVectorSecondary'] = raw_dataset["Game Rotation Vector -Wakeup Secondary"]
refined_dataset['LinearAcceleration'] = raw_dataset["Linear Acceleration"]
#refined_dataset['LinearAccelerationSecondary'] = raw_dataset["Linear Acceleration -Wakeup Secondary"]
refined_dataset['RotationVector'] = raw_dataset["Rotation Vector"]
# refined_dataset['RotationVectorSecondary'] = raw_dataset["Rotation Vector -Wakeup Secondary"]
refined_dataset['CoarseMotionClassifier'] = raw_dataset["Coarse Motion Classifier"]
refined_dataset['Pedometer'] = raw_dataset["Pedometer"]


refined_dataset.head()
refined_dataset = refined_dataset.fillna(0)


refined_dataset["StepCounter"] = refined_dataset['StepCounter'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset["WristTiltGesture"] = refined_dataset['WristTiltGesture'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset["HeartRateMonitor"] = refined_dataset['HeartRateMonitor'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset["StepCounterSecondary"] = refined_dataset['StepCounterSecondary'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset["GameRotationVector"] = refined_dataset["GameRotationVector"].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
#vector["GameRotationVectorSecondary"] = refined_dataset["GameRotationVectorSecondary"].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset['StepDetector'] = refined_dataset['StepDetector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset['RotationVector'] = refined_dataset['RotationVector'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
#refined_dataset['RotationVectorSecondary'] = refined_dataset['RotationVectorSecondary'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset['SignificantMotionDetector'] = refined_dataset['SignificantMotionDetector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset['LinearAcceleration'] = refined_dataset['LinearAcceleration'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset['Time'] = pd.to_datetime(refined_dataset.Time, format="%H:%M:%S.%f").apply(lambda x: x.time())



'''
Late November Data
'''
# single values based variables
refined_dataset_new['Date'] = raw_dataset_new["Date"]
refined_dataset_new['Time'] = raw_dataset_new["Time"]
refined_dataset_new['Trace'] = raw_dataset_new["TraceTime"]
refined_dataset_new['Quit'] = raw_dataset_new["QuitClicked"]
refined_dataset_new['Start'] = raw_dataset_new["StartClicked"]
refined_dataset_new['Setting'] = raw_dataset_new["SettingClicked"]
# single values based variables but stored in a single element list
refined_dataset_new['WristTiltGesture'] = raw_dataset_new["Wrist Tilt Gesture"]
refined_dataset_new['HeartRateMonitor'] = raw_dataset_new["Heart Rate Monitor -Wakeup Secondary"]
refined_dataset_new['StepCounter'] = raw_dataset_new["Step Counter"]
refined_dataset_new['StepCounterSecondary'] = raw_dataset_new["Step Counter -Wakeup Secondary"]
refined_dataset_new['StepDetector'] = raw_dataset_new["Step Detector"]
#refined_dataset_new['StepDetectorSecondary'] = raw_dataset_new["Step Detector -Wakeup Secondary"]
refined_dataset_new['SignificantMotionDetector'] = raw_dataset_new["Significant Motion Detector"]
refined_dataset_new['1000'] = raw_dataset_new["1000"]
refined_dataset_new['1001'] = raw_dataset_new["1001"]
refined_dataset_new['1002'] = raw_dataset_new["1002"]

# vector based variables
refined_dataset_new['RMD'] = raw_dataset_new["RMD"]
refined_dataset_new['GameRotationVector'] = raw_dataset_new["Game Rotation Vector"]
#refined_dataset_new['GameRotationVectorSecondary'] = raw_dataset_new["Game Rotation Vector -Wakeup Secondary"]
refined_dataset_new['LinearAcceleration'] = raw_dataset_new["Linear Acceleration"]
#refined_dataset_new['LinearAccelerationSecondary'] = raw_dataset_new["Linear Acceleration -Wakeup Secondary"]
refined_dataset_new['RotationVector'] = raw_dataset_new["Rotation Vector"]
# refined_dataset_new['RotationVectorSecondary'] = raw_dataset_new["Rotation Vector -Wakeup Secondary"]
refined_dataset_new['CoarseMotionClassifier'] = raw_dataset_new["Coarse Motion Classifier"]
refined_dataset_new['Pedometer'] = raw_dataset_new["Pedometer"]


refined_dataset_new.head()
refined_dataset_new = refined_dataset_new.fillna(0)

refined_dataset_new["StepCounter"] = refined_dataset_new['StepCounter'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset_new["WristTiltGesture"] = refined_dataset_new['WristTiltGesture'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset_new["HeartRateMonitor"] = refined_dataset_new['HeartRateMonitor'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset_new["StepCounterSecondary"] = refined_dataset_new['StepCounterSecondary'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset_new["GameRotationVector"] = refined_dataset_new["GameRotationVector"].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
#refined_dataset_new["GameRotationVectorSecondary"] = refined_dataset["GameRotationVectorSecondary"].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset_new['StepDetector'] = refined_dataset_new['StepDetector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset_new['RotationVector'] = refined_dataset_new['RotationVector'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
#refined_dataset_new['RotationVectorSecondary'] = refined_dataset_new['RotationVectorSecondary'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset_new['SignificantMotionDetector'] = refined_dataset_new['SignificantMotionDetector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset_new['LinearAcceleration'] = refined_dataset_new['LinearAcceleration'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset_new['Time'] = pd.to_datetime(refined_dataset_new.Time, format="%H:%M:%S.%f").apply(lambda x: x.time())

'''
July Data
'''

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
    #refined_df['StepDetectorSecondary'] = raw_df["Step Detector -Wakeup Secondary"]
    #refined_df['SignificantMotionDetector'] = raw_df["Significant Motion Detector"]
    refined_df['1000'] = raw_dataset["1000"]
    refined_df['1001'] = raw_dataset["1001"]
    refined_df['1002'] = raw_dataset["1002"]


    # vector based variables
    refined_df['RMD'] = raw_df["RMD"]
    refined_df['GameRotationVector'] = raw_df["Game Rotation Vector"]
    #refined_df['GameRotationVectorSecondary'] = raw_df["Game Rotation Vector -Wakeup Secondary"]
    refined_df['LinearAcceleration'] = raw_df["Linear Acceleration"]
    #refined_df['LinearAccelerationSecondary'] = raw_df["Linear Acceleration -Wakeup Secondary"]
    refined_df['RotationVector'] = raw_df["Rotation Vector"]
    # refined_df['RotationVectorSecondary'] = raw_df["Rotation Vector -Wakeup Secondary"]
    refined_df['CoarseMotionClassifier'] = raw_df["Coarse Motion Classifier"]
    refined_df['Pedometer'] = raw_df["Pedometer"]
    return(refined_df)


refined_dataset_july = data_extraction(raw_dataset_july, refined_dataset_july)

refined_dataset_july = refined_dataset_july.fillna(0)

def parse_vectors(refined_dataset):
    refined_dataset["StepCounter"] = refined_dataset['StepCounter'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset["HeartRateMonitor"] = refined_dataset['HeartRateMonitor'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset["StepCounterSecondary"] = refined_dataset['StepCounterSecondary'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset["GameRotationVector"] = refined_dataset["GameRotationVector"].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
    #vector["GameRotationVectorSecondary"] = refined_dataset["GameRotationVectorSecondary"].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
    refined_dataset['StepDetector'] = refined_dataset['StepDetector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset['RotationVector'] = refined_dataset['RotationVector'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
    #refined_dataset['RotationVectorSecondary'] = refined_dataset['RotationVectorSecondary'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
    #refined_dataset['SignificantMotionDetector'] = refined_dataset['SignificantMotionDetector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
    refined_dataset['LinearAcceleration'] = refined_dataset['LinearAcceleration'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
    refined_dataset['Time'] = pd.to_datetime(refined_dataset.Time, format="%H:%M:%S.%f").apply(lambda x: x.time())
    return(refined_dataset)
    
refined_dataset_july = parse_vectors(refined_dataset_july)
refined_dataset_july["WristTiltGesture"] = refined_dataset_july['WristTiltGesture'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))


'''
start_indices = np.where(refined_dataset["Start"] == 1)
quit_indices = np.where(refined_dataset["Quit"] == 1)
setting_indices = np.where(refined_dataset["Setting"] == 1)
start_index = [item.tolist() for item in start_indices][0]
quit_index = [item.tolist() for item in quit_indices][0]
setting_index = [item.tolist() for item in setting_indices][0]
'''


from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

'''
11/07 Data
'''

subset_1107 = refined_dataset[refined_dataset['Date'] == '2019-11-07']
subset_1107_08 = subset_1107[subset_1107['Time'] < pd.to_datetime('08:00:00.000').time()]
subset_1107_08['StepCounter'] = subset_1107_08['StepCounter'].replace(0, np.nan)
subset_1107_08['StepCounter'].fillna(method = 'bfill', inplace = True)
subset_1107_08['StepCounter'].fillna(0, inplace = True)

'''
11/15 Data
'''
subset_1115 = refined_dataset_new[refined_dataset_new['Date'] == '2019-11-15']
#subset_1115['StepCounter'] = subset_1115['StepCounter'].replace(0, np.nan)
subset_1115['StepCounter'].fillna(method = 'bfill', inplace = True)
#subset_1115['StepCounter'].fillna(0, inplace = True)

'''
11/09 Data
'''
subset_1109 = refined_dataset[refined_dataset['Date'] == '2019-11-09']
#subset_1109 = subset_1109[subset_1107['Time'] < pd.to_datetime('08:00:00.000').time()]
#subset_1109['StepCounter'] = subset_1109['StepCounter'].replace(0, np.nan)
subset_1109['StepCounter'].fillna(method = 'ffill', inplace = True)
#subset_1109['StepCounter'].fillna(0, inplace = True)



#plt.plot_date(subset_1107_08['Time'], subset_1107_08['StepCounter'], linestyle = '-')
plt.step(subset_1115['Time'], subset_1115['StepCounter'])

#plt.plot_date(subset_1107_08['Time'], subset_1107_08['HeartRateMonitor'], linestyle = '-')
plt.step(subset_1115['Time'], subset_1115['HeartRateMonitor'])


# extract data out from the list
vector_1115 = pd.DataFrame()
LinearAcceleration = subset_1115['LinearAcceleration'].apply(pd.Series)
LinearAcceleration = LinearAcceleration.rename(columns = lambda x : 'linacc_' + str(x))
vector_1115 = pd.concat([vector_1115, LinearAcceleration])
vector_1115.fillna(0, inplace = True)
vector_1115['Time'] = subset_1115['Time']

vector_1109 = pd.DataFrame()
LinearAcceleration = subset_1109['LinearAcceleration'].apply(pd.Series)
LinearAcceleration = LinearAcceleration.rename(columns = lambda x : 'linacc_' + str(x))
vector_1109 = pd.concat([vector_1109, LinearAcceleration])
vector_1109.fillna(0, inplace = True)
vector_1109['Time'] = subset_1109['Time']

vector_1107 = pd.DataFrame()
LinearAcceleration = subset_1107['LinearAcceleration'].apply(pd.Series)
LinearAcceleration = LinearAcceleration.rename(columns = lambda x : 'linacc_' + str(x))
vector_1107 = pd.concat([vector_1107, LinearAcceleration])
vector_1107.fillna(0, inplace = True)
vector_1107['Time'] = subset_1107['Time']


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



# calculate summary statistics
vector_1115.describe()
linaccmean = vector_1115.describe().mean()


plt.plot_date(vector_1115['Time'], vector_1115['linacc_0'], color = 'blue', linestyle = '-' )
plt.plot_date(vector_1115['Time'], vector_1115['linacc_1'], color = 'green', linestyle = '-' )
plt.plot_date(vector_1115['Time'], vector_1115['linacc_2'], color = 'red', linestyle = '-' )
plt.legend()

plt.step(vector_1115['Time'], vector_1115['linacc_0'], color = 'blue', linestyle = '-' )
plt.step(vector_1115['Time'], vector_1115['linacc_1'], color = 'green', linestyle = '-' )
plt.step(vector_1115['Time'], vector_1115['linacc_2'], color = 'red', linestyle = '-' )
plt.legend()

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,15))

g1 = fig.add_subplot(axes[0,0])
g2 = fig.add_subplot(axes[0,1])
g3 = fig.add_subplot(axes[0,2])
g4 = fig.add_subplot(axes[1,0])
g5 = fig.add_subplot(axes[1,1])
g6 = fig.add_subplot(axes[1,2])
g7 = fig.add_subplot(axes[2,0])
g8 = fig.add_subplot(axes[2,1])
g9 = fig.add_subplot(axes[2,2])
g10 = fig.add_subplot(axes[3,0])
g11 = fig.add_subplot(axes[3,1])
g12 = fig.add_subplot(axes[3,2])

g1.step(subset_1107['Time'], subset_1107['StepCounter'])
g2.step(subset_1109['Time'], subset_1109['StepCounter'])
g3.step(subset_1115['Time'], subset_1115['StepCounter'])
g4.step(subset_1107['Time'], subset_1107['HeartRateMonitor'])
g5.step(subset_1109['Time'], subset_1109['HeartRateMonitor'])
g6.step(subset_1115['Time'], subset_1115['HeartRateMonitor'])
g7.step(vector_1107['Time'], vector_1107['linacc_0'], color = 'blue', linestyle = '-' )
g7.step(vector_1107['Time'], vector_1107['linacc_1'], color = 'green', linestyle = '-' )
g7.step(vector_1107['Time'], vector_1107['linacc_2'], color = 'red', linestyle = '-' )
g8.step(vector_1109['Time'], vector_1109['linacc_0'], color = 'blue', linestyle = '-' )
g8.step(vector_1109['Time'], vector_1109['linacc_1'], color = 'green', linestyle = '-' )
g8.step(vector_1109['Time'], vector_1109['linacc_2'], color = 'red', linestyle = '-' )
g9.step(vector_1115['Time'], vector_1115['linacc_0'], color = 'blue', linestyle = '-' )
g9.step(vector_1115['Time'], vector_1115['linacc_1'], color = 'green', linestyle = '-' )
g9.step(vector_1115['Time'], vector_1115['linacc_2'], color = 'red', linestyle = '-' )
plt.legend()
g10.step(rot_vector_1107['Time'], rot_vector_1107['vec_0'], color = 'blue', linestyle = '-' )
g10.step(rot_vector_1107['Time'], rot_vector_1107['vec_1'], color = 'green', linestyle = '-' )
g10.step(rot_vector_1107['Time'], rot_vector_1107['vec_2'], color = 'red', linestyle = '-' )
g10.step(rot_vector_1107['Time'], rot_vector_1107['vec_3'], color = 'orange', linestyle = '-' )
g11.step(rot_vector_1109['Time'], rot_vector_1109['vec_0'], color = 'blue', linestyle = '-' )
g11.step(rot_vector_1109['Time'], rot_vector_1109['vec_1'], color = 'green', linestyle = '-' )
g11.step(rot_vector_1109['Time'], rot_vector_1109['vec_2'], color = 'red', linestyle = '-' )
g11.step(rot_vector_1109['Time'], rot_vector_1109['vec_3'], color = 'orange', linestyle = '-' )
g12.step(rot_vector_1115['Time'], rot_vector_1115['vec_0'], color = 'blue', linestyle = '-' )
g12.step(rot_vector_1115['Time'], rot_vector_1115['vec_1'], color = 'green', linestyle = '-' )
g12.step(rot_vector_1115['Time'], rot_vector_1115['vec_2'], color = 'red', linestyle = '-' )
g12.step(rot_vector_1115['Time'], rot_vector_1115['vec_3'], color = 'orange', linestyle = '-' )
plt.legend()

g1.title.set_text('11/07')
g2.title.set_text('11/09')
g3.title.set_text('11/15')
g1.set(ylabel = 'StepCounter')
g4.set(ylabel = 'HeartRate')
g7.set(ylabel = 'LinearAcceleration')
g10.set(ylabel = 'RotationVector')

plt.savefig('watchdata.png')


#g10.step(subset_1107['Time'], subset_1107['WristTiltGesture']) 
#g11.step(subset_1109['Time'], subset_1109['WristTiltGesture']) 
#g12.step(subset_1115['Time'], subset_1115['WristTiltGesture']) 



