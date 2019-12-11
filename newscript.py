#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:04:15 2019

@author: himanshu
"""

import pandas as pd
dataset = pd.read_csv("users.csv")
dataset.head()
refined_dataset = pd.DataFrame()
refined_dataset['Date'] = dataset["Date"]
refined_dataset['Time'] = dataset["Time"]
refined_dataset['Trace'] = dataset["TraceTime"]
refined_dataset['Quit'] = dataset["QuitClicked"]
refined_dataset['Start'] =dataset["StartClicked"]
refined_dataset['RMD'] = dataset["RMD"]
refined_dataset['Wrist Tilt Gesture'] = dataset["Wrist Tilt Gesture"]
refined_dataset['Heart Rate Monitor'] = dataset["Heart Rate Monitor -Wakeup Secondary"]
refined_dataset['GameRotationVector'] =dataset["Game Rotation Vector"]
#refined_dataset['GameRotationVectorSecondary'] = raw_dataset["Game Rotation Vector -Wakeup Secondary"]
refined_dataset['StepCounter'] = dataset["Step Counter"]
refined_dataset['StepCounterSecondary'] = dataset["Step Counter -Wakeup Secondary"]
refined_dataset['StepDetector'] = dataset["Step Detector"]
#refined_dataset['StepDetectorSecondary'] = raw_dataset["Step Detector -Wakeup Secondary"]
refined_dataset['LinearAcceleration'] = dataset["Linear Acceleration"]
#refined_dataset['LinearAccelerationSecondary'] = raw_dataset["Linear Acceleration -Wakeup Secondary"]
refined_dataset['RotationVector'] = dataset["Rotation Vector"]
refined_dataset['RotationVectorSecondary'] = dataset["Rotation Vector -Wakeup Secondary"]
refined_dataset['Significant Motion Detector'] = dataset["Significant Motion Detector"]
refined_dataset['Coarse Motion Classifier'] = dataset["Coarse Motion Classifier"]
refined_dataset['Pedometer'] = dataset["Pedometer"]
refined_dataset.head()
refined_dataset = refined_dataset.fillna(0) #fill na with zeros for averaging 
refined_dataset.head()
vector = pd.DataFrame()
refined_dataset["StepCounter"] = refined_dataset['StepCounter'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset["StepCounterSecondary"] = refined_dataset['StepCounterSecondary'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset["GameRotationVector"] = refined_dataset["GameRotationVector"].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
#vector["GameRotationVectorSecondary"] = refined_dataset["GameRotationVectorSecondary"].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset['StepDetector'] = refined_dataset['StepDetector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset['LinearAcceleration'] = refined_dataset['LinearAcceleration'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset['RotationVector'] = refined_dataset['RotationVector'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset['RotationVectorSecondary'] = refined_dataset['RotationVectorSecondary'].apply(lambda x: x if x == 0 else x.strip("[").strip("]").split(',')).apply(lambda x: x if x == 0  else list(map(float, x)))
refined_dataset['Significant Motion Detector'] = refined_dataset['Significant Motion Detector'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))
refined_dataset.head()
​
import numpy as np
start = np.where(refined_dataset["Start"] == 1)
end = np.where(refined_dataset["Quit"] == 1)
​
start = [item.tolist() for item in start][0]
end = [item.tolist() for item in end][0]
​
print(start)
print(end)
​
list_compressedDfs = []
for i in range(len(start)):
    temp = refined_dataset[end[i] : start[i] +1]
    temp["avgStepCounter"] = temp["StepCounter"].mean()
    #average columns in c
    #note need to do same as above for all scalar values functions, vector valued??
    list_compressedDfs.append(temp)
​
compressed = pd.concat(list_compressedDfs)
compressed.head()

subset_861_899= refined_dataset.iloc[861:900]
subset_first= subset_861_899.iloc[:,[8]]
list_values=[]
import numpy as np
for i in range(len(subset_first)):
     list_values.append(np.array(subset_first.iloc[i]['GameRotationVector']))
    


