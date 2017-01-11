
# coding: utf-8

# In[1]:

import pylab
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys
#%matplotlib inline
traffic_flow = pd.read_csv(sys.argv[1]+"/flow.tsv",
                             low_memory=False,sep = '\t', names = ['lane1','lane2','lane3'])
prob_flow=pd.read_csv(sys.argv[1]+"/prob.tsv",
                             low_memory=False,sep = '\t', names = ['m3_p1','m3_p2','m3_p3'])
m3_df=pd.concat([traffic_flow,prob_flow],axis=1)
m3_df.columns= ['m3_f1','m3_f2','m3_f3','m3_p1','m3_p2','m3_p3']
feature_data = traffic_flow.convert_objects(convert_numeric=True)
feature_data=feature_data[((feature_data["lane1"] >0) & (feature_data["lane1"] <=100))& ((feature_data["lane2"] >0) & (feature_data["lane2"] <=100)) & ((feature_data["lane3"] >0) & (feature_data["lane3"] <=100))]
train_data=feature_data[:900000]
test_data=feature_data[900000:]

feature_set_train=train_data.drop(["lane2"],axis=1)
feature_set_test=test_data.drop(["lane2"],axis=1)

regr = linear_model.LinearRegression()
regr.fit(feature_set_train, train_data["lane2"])
slope=max(regr.coef_)
intercept=regr.intercept_

predicted_lane_1=[]
predicted_lane_2=[]
predicted_lane_3=[]
p1=0.0
predictedDf=[6*[0] for i in range(len(traffic_flow))]
for i,row in enumerate(m3_df.itertuples()):
    prob_lane_1=row[4]
    prob_lane_2=row[5]
    prob_lane_3=row[6]
    if((float(row[2])>=0.0) & (float(row[3])>=0.0)):
        diff_1_2=abs(row[1]-row[2])
        diff_1_3=abs(row[1]-row[3])
        if(diff_1_2 <=diff_1_3):
            flow_lane_1=(slope*row[2])+intercept
            p1=prob_lane_2
        else:
            flow_lane_1=(slope*row[3])+intercept
            p1=prob_lane_3
    elif((float(row[2])<0.0) & (float(row[3])<0.0)):
        flow_lane_1=row[1]
        p1=prob_lane_1
    elif(float(row[2])>float(row[3])):
        flow_lane_1=(slope*float(row[2]))+intercept
        p1=prob_lane_2
    else:
        flow_lane_1=(slope*float(row[3]))+intercept
        p1=prob_lane_3
        
    if(float(row[1])<=0.0):
        flow_lane_2=flow_lane_1
        p2=prob_lane_1
    else:
        flow_lane_2=(slope*row[1])+intercept
        
    if(float(row[2])<=0.0):
        flow_lane_3=flow_lane_2
    else:
        flow_lane_3=(slope*row[2])+intercept
    predictedDf[i]=[flow_lane_1,flow_lane_2,flow_lane_3,p1,prob_lane_1,prob_lane_2]


# In[2]:

import math
from numpy import genfromtxt

flow_data = genfromtxt('/home/datascience/Lab10/cleaning2/1160/flow.tsv', delimiter='\t')
confidence_data = genfromtxt('/home/datascience/Lab10/cleaning2/1160/prob.tsv', delimiter='\t')
traffic_timeStamp = pd.read_csv("/home/datascience/Lab10/cleaning2/1160/timestamp.tsv",
                             low_memory=False,sep = '\t',parse_dates=[0], infer_datetime_format=True, names = ['timeStamp'])
temp = pd.DatetimeIndex(traffic_timeStamp['timeStamp'])
flow_date = temp.date
flow_time = temp.time
del traffic_timeStamp
totalNoLanes = len(flow_data[0])
totalNoOfRows = len(flow_data)


pred_flow = [[np.nan for x in range(totalNoLanes)] for y in range(totalNoOfRows)] 
method2_prob = [[np.nan for x in range(totalNoLanes)] for y in range(totalNoOfRows)] 
nanRemovedFlows = np.nan_to_num(flow_data)
nanRemovedProb = np.nan_to_num(confidence_data)


c_prev = 0
c_next = 0
for laneNo in range(totalNoLanes):
    for rowNo in range(1,totalNoOfRows - 1):
        if(pd.isnull(flow_data[rowNo][laneNo]) == False):
            if (flow_time[rowNo].minute - flow_time[rowNo -1].minute) <= 5 and flow_date[rowNo] == flow_date[rowNo -1]:
                c_prev = nanRemovedProb[rowNo - 1][laneNo]
            else:
                c_prev = 0
            
            if (flow_time[rowNo + 1].minute - flow_time[rowNo].minute) <= 5 and flow_date[rowNo] == flow_date[rowNo+1]:
                c_next = nanRemovedProb[rowNo + 1][laneNo]
            else:
                c_next = 0
            
            if (c_prev + c_next)!= 0:
                w_prev = (c_prev)/(c_prev + c_next)
                w_next = 1-w_prev
                pred_flow[rowNo][laneNo] = nanRemovedFlows[rowNo -1][laneNo] * w_prev + nanRemovedFlows[rowNo + 1][laneNo] * w_next
                method2_prob[rowNo][laneNo] = min(c_prev,c_next)
            else:
                pred_flow[rowNo][laneNo] = nanRemovedFlows[rowNo + 1][laneNo]
                method2_prob[rowNo][laneNo] = 0
                
                
for laneNo in range(totalNoLanes):
    pred_flow[0][laneNo] = nanRemovedFlows[1][laneNo]
    method2_prob[0][laneNo] = nanRemovedProb[1][laneNo]
    pred_flow[totalNoOfRows-1][laneNo] = nanRemovedFlows[totalNoOfRows -2][laneNo]
    method2_prob[totalNoOfRows-1][laneNo] = nanRemovedProb[totalNoOfRows -2][laneNo]


# In[3]:

del flow_data
del traffic_flow
del confidence_data
del prob_flow


# In[4]:

method_2_df=np.concatenate((pred_flow,method2_prob), axis=1)


# In[5]:

del pred_flow
del method2_prob
ensembleDf=np.concatenate((np.array(predictedDf),method_2_df,m3_df.as_matrix()), axis=1)


# In[6]:

del predictedDf
del method_2_df
del m3_df
ensemble_df=[18*[0] for i in range(ensembleDf.shape[0])]
for index,row in enumerate(ensembleDf):
    c1_1=row[3]
    c2_1=row[9]
    c3_1=row[15]
    if(float(c1_1+c2_1+c3_1) != 0.0):
        w1_1=(float(c1_1))/float(c1_1+c2_1+c3_1)
        w2_1=(float(c2_1))/float(c1_1+c2_1+c3_1)
        w3_1=(float(c3_1))/float(c1_1+c2_1+c3_1)
        predicted_flow_1 = (w1_1*row[0])+(w2_1*row[6])+(w3_1*row[12])
    else:
        predicted_flow_1=0
    
    c1_2=row[4]
    c2_2=row[10]
    c3_2=row[16]
    if(float(c1_2+c2_2+c3_2) != 0.0):
        w1_2=(float(c1_2))/float(c1_2+c2_2+c3_2)
        w2_2=(float(c2_2))/float(c1_2+c2_2+c3_2)
        w3_2=(float(c3_2))/float(c1_2+c2_2+c3_2)
        predicted_flow_2 = (w1_2*row[1])+(w2_2*row[7])+(w3_2*row[13])
    else:
        predicted_flow_2=0
    
    c1_3=row[5]
    c2_3=row[11]
    c3_3=row[17]
    if(float(c1_3+c2_3+c3_3) != 0.0):
        w1_3=(float(c1_3))/float(c1_3+c2_3+c3_3)
        w2_3=(float(c2_3))/float(c1_3+c2_3+c3_3)
        w3_3=(float(c3_3))/float(c1_3+c2_3+c3_3)
        predicted_flow_3 = (w1_3*row[2])+(w2_3*row[8])+(w3_3*row[14])
    else:
        predicted_flow_3=0
    ensemble_df[index]=[int(predicted_flow_1),int(predicted_flow_2),int(predicted_flow_3)]


# In[7]:

ensemble_df=np.array(ensemble_df)


# In[17]:

np.savetxt("./1160.flow.txt",ensemble_df, delimiter="\t",fmt='%i')


# In[ ]:



