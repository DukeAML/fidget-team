#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:55:17 2019

@author: himanshu
"""

import pandas as pd
raw_dataset= pd.read_csv('set.csv')
raw_dataset.head()


refined_dataset = pd.DataFrame()
refined_dataset['Date'] = raw_dataset["Date"]
refined_dataset['Time'] = raw_dataset["Time"]
refined_dataset['Trace'] = raw_dataset["TraceTime"]
refined_dataset['Quit'] = raw_dataset["QuitClicked"]
refined_dataset['Start'] = raw_dataset["StartClicked"]
refined_dataset['Coarse Motion Classifier'] = raw_dataset["Coarse Motion Classifier"]
refined_dataset['Pedometer'] = raw_dataset["Pedometer"]
refined_dataset['RMD'] = raw_dataset["RMD"]
refined_dataset['Wrist Tilt Gesture'] = raw_dataset["Wrist Tilt Gesture"]
refined_dataset['Heart Rate Monitor'] = raw_dataset["Heart Rate Monitor -Wakeup Secondary"]

refined_dataset['Game Rotation Vector'] = raw_dataset["Game Rotation Vector -Wakeup Secondary"]
refined_dataset['Step Counter'] = raw_dataset["Step Counter -Wakeup Secondary"]
refined_dataset['Step Detector'] = raw_dataset["Step Detector -Wakeup Secondary"]
refined_dataset['Linear Acceleration'] = raw_dataset["Linear Acceleration -Wakeup Secondary"]
refined_dataset['Rotation Vector'] = raw_dataset["Rotation Vector -Wakeup Secondary"]
refined_dataset['Significant Motion Detector'] = raw_dataset["Significant Motion Detector"]
refined_dataset['x?'] = raw_dataset["tsl258x-light"] #what is this?

#do we need to unpack these click values from another column?
# refined_dataset['Next'] = raw_dataset["Next Clicked"]
# refined_dataset['Menu'] = raw_dataset["Menu Clicked"]
# refined_dataset['Happy'] = raw_dataset["Happy Clicked"]
# refined_dataset['Sad'] = raw_dataset["Sad Clicked"]
# refined_dataset['Neutral'] = raw_dataset["Neutral Clicked"]
# refined_dataset['Setting'] = raw_dataset["Setting Clicked"]
# refined_dataset['Exit'] = raw_dataset["Exit Clicked"]

refined_dataset.head()
#maybe want to introduce some raise errors for unreasonable values 
refined_dataset = refined_dataset.fillna(0) #fill na with zeros for averaging 
refined_dataset.head()


refined_dataset["Step Counter"] = refined_dataset['Step Counter'].apply(lambda x: x if x == 0 else int(x.strip("[").strip("]")))


refined_dataset.head()


import numpy as np
start_indices = np.where(refined_dataset["Start"] == 1)
indices = np.where(refined_dataset["Quit"] == 1)
​
start = [item.tolist() for item in start_indices][0]
end = [item.tolist() for item in indices][0]
​
print(start)
print(end) #seems like we should be indexing quit-to-start ?

list_compressedDfs = []

for ii in range(len(start)):
    temp = refined_dataset[end[ii] : start[ii] +1]
    temp["avgStepCounter"] = temp["Step Counter"].mean()
    #average columns in c
    #note need to do same as above for all scalar values functions, vector valued??
    list_compressedDfs.append(temp)
    
compressed = pd.concat(list_compressedDfs)

compressed.head()
