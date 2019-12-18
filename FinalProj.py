#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:03:33 2019

@author: ds
"""
###### INSTALL PACKAGES ###########
#pip install hyperloglog
#pip install pyprobables
#pip install pandas_ml
#############   ###################
import csv
import json
import pandas as pd
#import codecs
import numpy as np
import numpy.linalg as la
import seaborn as sns
import matplotlib.pyplot as plt
#import MySQLdb
import sqlite3
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from math import sqrt
from probables import CountMinSketch
from sklearn.metrics import mean_squared_error
import hyperloglog
from pandas_ml import ConfusionMatrix

#############################################################
##################### CONFIGURATION #########################
#############################################################
 
#Test quickTest =1 / Full Run quickTest =0
quickTest = 0
#PART A Limit Users:
limitA = 15
#PART B Limit rows
limitB = 2000

#OS:
linux  = 0
windows = 1

########  PART A parameters  ######

#Similarity Thresholds
simThreshForFriend = 0.2
simThreshForNeighbor = 0.1
#k-nearest neighbours threshold
kNearNighbThres = 4

########  PART B parameters  ######

#CountMinSketch depth, width
depth=300
width=750
#HyperLogLog Error
errorHll = 0.01
#Sampling
sampleSize = 50

#Data Source File
if linux == 1:
    path = '~/Final_Project/hetrec2011-lastfm-2k/' 
    storeResults = '~/Final_Project/'
    #Encoding
    enc = 'ISO-8859-15'

if windows == 1:
    path = 'C:/Users/C5267840/OneDrive - SAP SE/Deree/Courses/Introduction to Big Data/Final_Proj/'
    storeResults = 'C:/Users/C5267840/OneDrive - SAP SE/Deree/Courses/Introduction to Big Data/Final_Proj/'
    #Encoding
    enc = 'mbcs'
 
##################  Enable Questions to run!!! <don't run> = 0 / <run> = 1  #################

#######  PART A - Question 1  ########################
AQ1 = 0

# 1. Create lastFM schema
AQ2_FndSims_1 = 0

# 2. User similarity 
AQ2_FndSims_2 = 0

# 2a. User similarity  results to csv 
AQ2_FndSims_2a = 0

# 2b. User similarity  results to database
AQ2_FndSims_2b = 0

########  PART A - Question 2
# 3. The k-nearest neighbours per user
AQ2_FndSims_3 = 0

# 3a. The k-nearest neighbours per user to JSON 
AQ2_FndSims_3a = 0

# 3b. The k-nearest neighbours per user to database 
AQ2_FndSims_3b = 0

#Reccomend #Evaluate #Improve
AQ2_Rcmd_1 = 0

 
######## PART B - Question 1: Heavy hitters  ############
BQ1_1 = 0
BQ1_2 = 0
BQ1_3 = 0

######## PART B - Question 2: Counting unique
BQ2_1 = 1
BQ2_2 = 1

######## PART B - Question 3: Streaming processing
BQ3 = 1

##########################################################################################
##########################################################################################

#Artists 
fArt = path + 'artists.dat'
dArt = pd.read_csv(fArt, encoding='utf-8', delimiter='\t') 
dfArt= pd.DataFrame(dArt)

#Tags 
fTags = path + 'tags.dat' 
dTags = pd.read_csv(fTags, header=0 , encoding= enc, delimiter='\t') 
dfTags = pd.DataFrame(dTags)

#User Artists 
fUsArt = path + 'user_artists.dat'
dUsArt= pd.read_csv(fUsArt, delimiter='\t') 
dfUsArt = pd.DataFrame(dUsArt)

#User Tagged Artists 
fUsTagArt = path + 'user_taggedartists.dat'
dUsTagArt = pd.read_csv(fUsTagArt, delimiter='\t') 
dfUsTagArt = pd.DataFrame(dUsTagArt)

#User Tagged Artists Timestamps 
fUsTagArtTim = path + 'user_taggedartists-timestamps.dat'
dUsTagArtTim = pd.read_csv(fUsTagArtTim, delimiter='\t')
dfUsTagArtTim = pd.DataFrame(dUsTagArtTim)

#User Friends 
fUsFrds = path + 'user_friends.dat'
dUsFrds = pd.read_csv(fUsFrds, delimiter='\t') 
dfUsFrds= pd.DataFrame(dUsFrds)

#PART A - Question 1
if AQ1 == 1:
    #Artists Listening Frequency 
    dfUsArtLFrq = dfUsArt.groupby(['userID']).sum()
    plt.figure(figsize=(40,20))
    count , bins, ignored= plt.hist(dfUsArtLFrq.weight, 10, normed=False)
    plt.show()
    #Frequency of Tags per Users
    dfTagUsFrq = dfUsTagArt.groupby(['userID']).count()
    plt.figure(figsize=(40,20))
    count , bins, ignored= plt.hist(dfTagUsFrq.tagID, 10, normed=False)
    plt.show()
    #Frequency of Tags per Artists
    dfTagArtFrq = dfUsTagArt.groupby(['artistID']).count()
    plt.figure(figsize=(40,20))
    count , bins, ignored= plt.hist(dfTagArtFrq.tagID, 10, normed=False)
    plt.show()

    #Outlier Detection
    #For 3 std, 99.7%, outliers of biggest values
    #User outliers
    stdUs = round(dfTagUsFrq.tagID.std())
    userOutlTag = dfTagUsFrq[dfTagUsFrq.tagID > 3*stdUs]
    userOutlTag.rename(columns={"tagID": "NumberOfTags"}, inplace=True) 
    print('The user outliers per tag are:')
    print(userOutlTag.NumberOfTags)
    
    
    #Tag ouliers
    dfUsTagFrq = dfUsTagArt.groupby(['tagID']).count()
    stdTag = round(dfUsTagFrq.userID.std())
    tagsOutlUs = dfUsTagFrq[dfUsTagFrq.userID > 3*stdTag]
    tagsOutlUs.rename(columns={"userID": "NumberOfUsers"}, inplace=True) 
    print('The user outliers per tag are:')
    print(tagsOutlUs.NumberOfUsers)

#PART A - Question 2 
# 1. Create lastFM schema
if AQ2_FndSims_1 == 1:
    #Create database schema lastFM
#    db = MySQLdb.connect(host="localhost",
#                         user="ds",
#                         passwd="0019cg!",
#                         unix_socket="/var/run/mysqld/mysqld.sock" )
#    
#    dbConnection=db.cursor()
#    
#    dbConnection.execute("drop database if exists lastFM")
#    dbConnection.execute("create database lastFM ")
#    dbConnection.execute("use lastFM")
#    dbConnection.close()
    sqlEngine       = create_engine('mysql+pymysql://ds:0019cg!@localhost/lastFM?charset=utf8&unix_socket=/var/run/mysqld/mysqld.sock', encoding='utf8', echo=True )
    
    dbConnection    = sqlEngine.connect() 
    dbConnection.execute("drop database if exists lastFM")
    dbConnection.execute("create database lastFM")
    dbConnection.execute("use lastFM")
    # Create artists table
    #Dynamically auto applying columns' length 10% larger than the max value
    sql1 = """CREATE TABLE artists (
             id  INT NOT NULL primary key,
             name  VARCHAR(""" + str(dfArt.name.apply(lambda x : len(x)).max()*1.1) + """),
             url VARCHAR(""" + str(dfArt.url.apply(lambda x : len(x)).max()*1.1) + """),  
             pictureURL VARCHAR(""" + str(dfArt.pictureURL.apply(lambda x : len(str(x))).max()*1.1) + """)
             )  ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci"""
    
    #Create tags table
    sql2 = """CREATE TABLE tags (
             tagID  INT NOT NULL primary key,
             tagValue  VARCHAR(""" + str(dfTags.tagValue.apply(lambda x : len(str(x))).max()*1.1) + """)
             )"""
    #create user_artists table
    sql3 = """CREATE TABLE user_artists (
             userID   INT,  
        artistID INT,
        weight INT,
             primary key (userID, artistID), 
             foreign key (artistID) references artists(id)
             )"""
    
    sql4 = """CREATE TABLE user_taggedartists (
             userID   INT,  
        artistID INT,
        tagID INT,
             day INT,
             month INT,
             year INT,
             primary key (userID, artistID, tagID) 
             )"""
    
    sql5 = """CREATE TABLE user_taggedartists_timestamps (
             userID   INT,  
        artistID INT,
        tagID INT,
             timestamp BIGINT,
             primary key (userID, artistID, tagID) 
             )"""
    
    sql6 = """CREATE TABLE user_friends (
             userID   INT,  
        friendID INT,
        primary key (userID, friendID), 
             foreign key (userID) references user_artists(userID)
             )"""
    
    sql7 = """CREATE TABLE users_pairs (
             userID_i   INT,  
        userID_j INT,
             sim DOUBLE,
        primary key (userID_i, userID_j) 
             )"""
    
    dbConnection.execute(sql1)
    dbConnection.execute(sql2)
    dbConnection.execute(sql3)
    dbConnection.execute(sql4)
    dbConnection.execute(sql5)
    dbConnection.execute(sql6)
    dbConnection.execute(sql7)


    tableArt = "artists"
    tableTags = "tags"
    tableUsArt = "user_artists"
    tableUsFrds = "user_friends"
    tableUsTagArt = "user_taggedartists"
    tableUsTagArtTim = "user_taggedartists_timestamps"
    try:
        statement = text("""SET NAMES UTF8""")
        dbConnection.execute(statement)
        frameArt  = dfArt.to_sql(tableArt, dbConnection, if_exists='append', index=False);
        frameTags  = dfTags.to_sql(tableTags, dbConnection, if_exists='append', index=False);
        frameUsArt  = dfUsArt.to_sql(tableUsArt, dbConnection, if_exists='append', index=False);
        frameUsFrds  = dfUsFrds.to_sql(tableUsFrds, dbConnection, if_exists='append', index=False);
        frameUsTagArt  = dfUsTagArt.to_sql(tableUsTagArt, dbConnection, if_exists='append', index=False);
        frameUsTagArtTim  = dfUsTagArtTim.to_sql(tableUsTagArtTim, dbConnection, if_exists='append', index=False); 
        
    except ValueError as vx: 
        print(vx) 
    except Exception as ex:   
        print(ex) 
    else: 
        print("Data loaded successfully.");    
    finally: 
        dbConnection.close() 

#PART A - Question 2 
# 2. User similarity       
if AQ2_FndSims_2 == 1:

    users = [] 
    users =  dfUsArt.userID.unique() 
    if quickTest == 1:
        users = users[:limitA]  
    usSim = pd.DataFrame( data=None, index=None, columns = None)
    
    i=0 
    j=0 
    arA=[] 
    arB=[] 
    dfUsArtP = pd.pivot_table(dfUsArt, index='userID', columns='artistID', values='weight') 
    dfUsArtP = dfUsArtP.fillna(0)
    
    while i <= (len(users) -1) in users:
    
        j = 0
     
        while j < len(users) in users:
            u_i=users[i]
            u_j=users[j]
            if u_i != u_j:
                arA = dfUsArtP.loc[u_i].values
        
                arB = dfUsArtP.loc[u_j].values
        
                sim = np.double(np.dot(arA, arB)/(la.norm(arA)*la.norm(arB))) 
                usSim = usSim.append( [[u_i, u_j, sim]],ignore_index=True)
     
            j = j+1

        i = i+1
    usSim.rename(columns={0: "userID_i"}, inplace=True) 
    usSim.rename(columns={1: "userID_j"}, inplace=True) 
    usSim.rename(columns={2: "sim"}, inplace=True) 
    
#PART A - Question 2 
# 2a. User similarity  results to csv 
if AQ2_FndSims_2a == 1:
    usSim.to_csv(storeResults + "usersSimilarity.csv") 

#PART A - Question 2 
# 2b. User similarity  results to database 
if AQ2_FndSims_2b == 1: 
    usSim.loc[:,'sim'] = usSim.sim.apply(lambda x : np.float64(x)  )
    sqlEngine = create_engine('mysql+pymysql://ds:0019cg!@localhost/lastFM?charset=utf8&unix_socket=/var/run/mysqld/mysqld.sock', encoding='utf8', echo=True )
    
    dbConnection = sqlEngine.connect() 
     
    try:
        statement = text("""SET NAMES UTF8""")
        dbConnection.execute(statement)
        frameUsPairs  = usSim.to_sql('users_pairs', dbConnection, if_exists='append', index=False);
         
        
    except ValueError as vx: 
        print(vx) 
    except Exception as ex:   
        print(ex) 
    else: 
        print("User-pairs data loaded successfully.");    
    finally: 
        dbConnection.close() 

#PART A - Question 2
# 3. The k-nearest neighbours per user
KNN = kNearNighbThres
if AQ2_FndSims_3 == 1:
 
    i=0
    k_near = pd.DataFrame( data=None, index=None, columns = None)
    js_near = pd.DataFrame( data=None, index=None, columns = None)
    while i < (len(users)) in users:
   
          n = 0
          us_i = users[i]
          nghb = usSim[usSim['userID_i']==us_i].sim.nlargest(KNN)
          nghb = nghb.reset_index()
          nghb.rename(columns={"index": "userID_j"}, inplace=True)
          for n , row in nghb.iterrows():
               us_j = nghb.loc[n].userID_j
               us_j = np.int(usSim.loc[us_j].userID_j)
               sim_k = nghb.loc[n].sim
               
               k_near = k_near.append( [[us_i, us_j, sim_k]],ignore_index=True)
#             
               n = n + 1
          i = i + 1
    k_near.rename(columns={0: "userID_i"}, inplace=True)
    k_near.rename(columns={1: "userID_j"}, inplace=True)
    k_near.rename(columns={2: "sim"}, inplace=True)
    k_near = k_near[k_near['sim']!=0]
#PART A - Question 2
# 3a. The k-nearest neighbours per user to JSON    
if AQ2_FndSims_3a == 1:          
    i=0
   
    k=0
    k_nearJSN = k_near[['userID_i' ,'userID_j']].groupby("userID_i").apply(lambda g: g.to_dict(orient='records'))
    while i < (len(k_nearJSN.index.values)):
        k =  k_nearJSN.index.values[i]
        j=0
        while j < (len(k_nearJSN[k])):
            us = k_nearJSN[k][j]['userID_j']
            k_nearJSN[k][j] = us
            j = j +1
        i = i+1
    k_nearJSN.to_json(storeResults + 'neighbors-k-lastFM.data')      

#PART A - Question 2
# 3b. The k-nearest neighbours per user to database  
if AQ2_FndSims_3b == 1:
    sqlEngine       = create_engine('mysql+pymysql://ds:0019cg!@localhost/lastFM?charset=utf8&unix_socket=/var/run/mysqld/mysqld.sock', encoding='utf8', echo=True )

    dbConnection    = sqlEngine.connect()  
   
    try:
        statement1 = text("""SET NAMES UTF8""")
        statement2 = text("""CREATE TABLE user_neighbors ( userID_i   INT,   userID_j INT,   primary key (userID_i, userID_j) )""")
        dbConnection.execute(statement1)
        dbConnection.execute(statement2)
        k_near = k_near[['userID_i' ,'userID_j']]
        frameUsNeib  = k_near.to_sql('user_neighbors', dbConnection, if_exists='append', index=False);
         
    except ValueError as vx:
        print(vx)
    except Exception as ex:  
        print(ex)
    else:
        print("User-neighbors data loaded successfully.");    
    finally:
        dbConnection.close()      
        
#PART A - Question 2  
#Reccomend #Evaluate #Improve
        
#Recomendation system from similar neighbors        
if AQ2_Rcmd_1 == 1:
    simThreshForNeighbor = simThreshForNeighbor
    #weight
    dfUsArtRecmd = dfUsArtP.copy(deep=True)
    dfUsArtRecmd[dfUsArtRecmd>0] = 0
    #binary
    dfUsArtRecmdBin = dfUsArtRecmd.copy(deep=True)
    k_nearPerUser = k_nearJSN.reset_index()
    k_nearPerUser.rename(columns={0: "userID_j"}, inplace=True)
    dfCmnArt = pd.DataFrame( data=None, index=None, columns = None)
    i=0
    #per focused user
    mae=0
    n=0
    total=0
    while i < (len(k_nearPerUser)):
        k_nearCluster = pd.DataFrame( data=None, index=None, columns = None)
        k_usrs = pd.DataFrame( data=None, index=None, columns = None)
        #focused user
        fUser = k_nearPerUser.loc[i]['userID_i'] 
        arUsArt = dfUsArtP.loc[fUser]
        
        #neighbors
        #Excluded focused User
        #k_usrs = k_usrs.append([fUser],ignore_index=True)
        #k_nearCluster = k_nearCluster.append([arUsArt],ignore_index=True) 
        k_neig = k_nearPerUser.loc[i]['userID_j']
        length = len(k_nearPerUser.loc[i]['userID_j']) 
        j=0
        while j < (length):
            k_neigUs = k_neig[j]
            arUsArt = dfUsArtP.loc[k_neigUs]
            k_usrs = k_usrs.append([k_neigUs],ignore_index=True)
            k_nearCluster = k_nearCluster.append( [arUsArt],ignore_index=True)
            k_nearClusterAll = k_usrs.join(k_nearCluster)
            j = j +1 
            
        #Common Artists of neighbors cluster
        cmnArt = k_nearClusterAll.loc[:, (k_nearClusterAll != 0).all()]
        cmnArtNum = len(cmnArt.columns) - 1
        dfCmnArt = dfCmnArt.append( [[fUser, cmnArtNum]],ignore_index=True)
        sumWS = 0
        allWgt = 0
        #1st column is user ID
        k=1
        while k <= (cmnArtNum):
            art = cmnArt.columns[k]
            g=1
            while g < (len(cmnArt)):
                
                cmnArtWgt = cmnArt.loc[g][art] 
                usr_j = cmnArt.loc[g][0]
                simUs = usSim.loc[ (usSim['userID_i']==fUser) & (usSim['userID_j']==usr_j) ].sim.values[0]
                
                #Improve with similarity Threshold for each neighbor
                if simUs > simThreshForNeighbor:
                    n=n+1
                    sumWS = sumWS + (simUs * cmnArtWgt)
                    allWgt = allWgt + simUs
                    #Reccomend 
                    artRecmd = sumWS/allWgt
                    dfUsArtRecmd.loc[fUser][art] = artRecmd
                    
                    #Evaluate recommend for at least 1 listening of a recommendation
                    #IF recommendation greater than 1
                    if artRecmd > 0:
                        dfUsArtRecmdBin.loc[fUser][art] = 1
                        rec = 1
                    else:
                        dfUsArtRecmdBin.loc[fUser][art] = 0 
                        rec = 0
                     #IF actual listening of artist from focused user greater than 1    
                    if dfUsArtP.loc[fUser][art] > 0:
                        act = 1
                    else:
                        act = 0 
                        
                    mae=mae+np.abs(rec-act)
                    diff = rec - act
                    total += diff * diff 

                g=g+1
            k=k+1
        i = i+1
    if n > 0:
        mae = mae/n
        rmse = sqrt(total / n)
        print("MAE for at least 1 listening of similar neighbor recommendation: ",mae)
        print("RMSE for at least 1 listening of similar neighbor recommendation: ",rmse)
    else:
        print("No common artists in neighbors for similarity threshold: ", simThreshForNeighbor)
#Improve
#Recomendation system from friends
    simThreshForFriend = simThreshForFriend
    
    i=0
   
    k=0
    if quickTest == 1:
        dfUsFrds = dfUsFrds[:10] 
    dfUsFrdsJSN = dfUsFrds[['userID' ,'friendID']].groupby("userID").apply(lambda g: g.to_dict(orient='records'))
    while i < (len(dfUsFrdsJSN.index.values)):
        k =  dfUsFrdsJSN.index.values[i]
        j=0
        while j < (len(dfUsFrdsJSN[k])):
            us = dfUsFrdsJSN[k][j]['friendID']
            dfUsFrdsJSN[k][j] = us
            j = j +1
        i = i+1
    #weight
    dfFrdUsArtRecmd = dfUsArtP.copy(deep=True)
    dfFrdUsArtRecmd[dfUsArtRecmd>0] = 0
    #binary
    dfFrdUsArtRecmdBin = dfFrdUsArtRecmd.copy(deep=True)
    k_frdsPerUser = dfUsFrdsJSN.reset_index()
    k_frdsPerUser.rename(columns={0: "friendID"}, inplace=True)
    dfCmnArtFrd = pd.DataFrame( data=None, index=None, columns = None)
    i=0
    #per focused user
    mae=0
    n=0
    total=0
    while i < (len(k_frdsPerUser)):
     
        k_friendsCluster = pd.DataFrame( data=None, index=None, columns = None)
        k_usrs = pd.DataFrame( data=None, index=None, columns = None)
        #focused user
        fUser = k_frdsPerUser.loc[i]['userID'] 
        arUsArt = dfUsArtP.loc[fUser]
        
        #friends cluster
        #k_usrs = k_usrs.append([fUser],ignore_index=True)
        #k_friendsCluster = k_friendsCluster.append([arUsArt],ignore_index=True) 
        k_neig = k_frdsPerUser.loc[i]['friendID']
        length = len(k_frdsPerUser.loc[i]['friendID']) 
        j=0
        while j < (length):
        
            k_neigUs = k_neig[j]
            arUsArt = dfUsArtP.loc[k_neigUs]
            k_usrs = k_usrs.append([k_neigUs],ignore_index=True)
            k_friendsCluster = k_friendsCluster.append( [arUsArt],ignore_index=True)
            k_friendsClusterAll = k_usrs.join(k_friendsCluster)
            j = j +1 
        #Common Artists
        cmnArt = k_friendsClusterAll.loc[:, (k_friendsClusterAll != 0).all()]
        cmnArtNum = len(cmnArt.columns) - 1
        dfCmnArt = dfCmnArt.append( [[fUser, cmnArtNum]],ignore_index=True)
        sumWS = 0
        allWgt = 0
        #1st column is user ID
        k=1
        while k <= (cmnArtNum):
     
            art = cmnArt.columns[k]
            g=1
            while g < (len(cmnArt)):
        
                cmnArtWgt = cmnArt.loc[g][art] 
                usr_j = cmnArt.loc[g][0]
                if len(usSim[(usSim['userID_i']==fUser) & (usSim['userID_j']==usr_j)].index.values) > 0:
                    simUs = usSim.loc[ (usSim['userID_i']==fUser) & (usSim['userID_j']==usr_j) ].sim.values[0]
                    #Improve with similarity Threshold
                    if simUs >= simThreshForFriend:
                        n=n+1
                        sumWS = sumWS + (simUs * cmnArtWgt)
                        allWgt = allWgt + simUs
                        #Reccomend 
                        artRecmd = sumWS/allWgt
                        dfFrdUsArtRecmd.loc[fUser][art] = artRecmd
                        
                        #Evaluate recommend for at least 1 listening of a recommendation
                        #IF recommendation greater than 1
                        if artRecmd > 0:
                            dfFrdUsArtRecmdBin.loc[fUser][art] = 1
                            rec = 1
                        else:
                            dfFrdUsArtRecmdBin.loc[fUser][art] = 0 
                            rec = 0
                         #IF actual listening of artist from focused user greater than 1    
                        if dfUsArtP.loc[fUser][art] > 0:
                            act = 1
                        else:
                            act = 0 
                        #Evaluate 
                        mae=mae+np.abs(rec-act)
                        diff = rec - act
                        total += diff * diff 

                g=g+1
            k=k+1
        i = i+1
    if n > 0:
        mae = mae/n
        rmse = sqrt(total / n)
        print("MAE for at least 1 listening of friends recommendation: ",mae)
        print("RMSE for at least 1 listening of friends recommendation: ",rmse)
    else:
        print("No common artists in friends for similarity threshold: ", simThreshForFriend)
    
#PART B  
fStrPr =  path + 'data-streaming-project.data' 
dStrPr = pd.read_csv(fStrPr, delimiter='\t',header=None, names=['user', 'movie',  'rating', 'timestamp']) 
dfStrPr = pd.DataFrame(dStrPr)  
 
if quickTest ==1:
    dfStrPr = dfStrPr[:limitB]
 
#Q1: Heavy hitters    
if BQ1_1 == 1:
    start =0
    end = 1000
    strmStep = 1000
    movsFrq = pd.DataFrame( data={'Frq': []}, index=None, columns=['Frq'])
    movsFrq.index.name='Movie'
    movsFrqSkt = pd.DataFrame( data={'Frq': []}, index=None, columns=['Frq'])
    movsFrqSkt.index.name='Movie'
#    min-sketch parameters
#    The width has an impact on the number of collisions while the depth on the approximation accuracy.
    depth=depth
    width=width
    cms = CountMinSketch (depth,width)
    n = 0
    while n < len(dfStrPr):
        dfStrPrWin = dfStrPr[start: end]
#        print("start:", start)
#        print("end:", end)
#        print(dfStrPrWin)
        for m , row in dfStrPrWin.iterrows():
            mov = dfStrPrWin.loc[m, "movie"]
            #BQ1_1 movie frq counter
            if mov in movsFrq.index:
                movsFrq.loc[mov]['Frq'] = movsFrq.loc[mov]['Frq'] + 1
            else:
                movsFrq.loc[mov] = [1]
            #BQ1_2 min-sketch
            if BQ1_2 == 1:
                mov_s = str(mov)
                cms.add(mov_s)
                movsFrqSkt.loc[mov] = cms.check(mov_s)
                
        start = start + strmStep
        end = end  + 1000 
        n =n + strmStep
    movsFrq.to_csv(storeResults + "movie-counter.csv")
    
    #BQ1_3 Compare method accuracy
    if BQ1_3 == 1:
        movsFrq = movsFrq.sort_values('Frq', ascending=False)
        movsFrqSkt = movsFrqSkt.sort_values('Frq', ascending=False)
 
        rmse = mean_squared_error(movsFrq, movsFrqSkt)
        print("BQ1_3 Compare method accuracy, RMSE: ",  rmse)

    
#Q2: Counting unique
#BQ2_1 Number of Unique movies & users  
if BQ2_1 == 1:
    start =0
    end = 1000
    strmStep = 1000
    winUniqueRcdA = pd.DataFrame( data={'unqMovies': [], 'unqUsers':[]}, index=None, columns=['unqMovies','unqUsers'])
    winUniqueRcdA.index.name='Window'
    winUniqueRcdB = pd.DataFrame( data={'unqMovies': [], 'unqUsers':[]}, index=None, columns=['unqMovies','unqUsers'])
    winUniqueRcdB.index.name='Window'
    movsFrqQ2 = pd.DataFrame( data={'Exists': []}, index=None, columns=['Exists'])
    movsFrqQ2.index.name='Movie'
    usrsFrqQ2 = pd.DataFrame( data={'Exists': []}, index=None, columns=['Exists'])
    usrsFrqQ2.index.name='User'
    winUniqueMoviesB = 0
    winUniqueUsersB  = 0
    win =0 
    n = 0 
    while n < len(dfStrPr):
        winN = 'Window_' + str(win) 
        dfStrPrWin = dfStrPr[start: end]
        #A - Simplest impementation
        #Movies
        winUniqueMoviesA = len(dfStrPrWin['movie'].unique()) 
        #Users
        winUniqueUsersA = len(dfStrPrWin['user'].unique())
        winUniqueRcdA.loc[winN] = [winUniqueMoviesA, winUniqueUsersA]
        #B - Simplest logic
        for m , row in dfStrPrWin.iterrows():
            #Movies
            mov = dfStrPrWin.loc[m, "movie"] 
            if mov in movsFrqQ2.index:
                winUniqueMoviesB = winUniqueMoviesB
            else:
                movsFrqQ2.loc[mov]  = ['True']
                winUniqueMoviesB = winUniqueMoviesB + 1
            #Users    
            usr = dfStrPrWin.loc[m, "user"] 
            if usr in usrsFrqQ2.index:
                winUniqueUsersB = winUniqueUsersB
            else:
                usrsFrqQ2.loc[usr] = ['True']
                winUniqueUsersB = winUniqueUsersB + 1
        winUniqueRcdB.loc[winN] = [winUniqueMoviesB, winUniqueUsersB]
        
        win = win + 1    
        start = start + strmStep
        end = end  + 1000 
        n =n + strmStep
    print("Bytes for Unique Movies Dataframe: ",  winUniqueRcdB['unqMovies'].__sizeof__())
    print("Bytes for Unique Users Dataframe: ",  winUniqueRcdB['unqUsers'].__sizeof__())
#BQ2_1 Number of Unique movies & users with HyperLogLog
if BQ2_2 == 1:   
    start =0
    end = 1000
    strmStep = 1000
    errorHll = errorHll
    winUniqueRcdC = pd.DataFrame( data={'unqMovies': [], 'unqUsers':[]}, index=None, columns=['unqMovies','unqUsers'])
    winUniqueRcdC.index.name='Window'
    hllMovies = hyperloglog.HyperLogLog(errorHll)
    hllUsers = hyperloglog.HyperLogLog(errorHll)
    win =0 
    n = 0
    while n < len(dfStrPr):
        winN = 'Window_' + str(win) 
        dfStrPrWin = dfStrPr[start: end]
        for m , row in dfStrPrWin.iterrows():
            #Movies
            mov = dfStrPrWin.loc[m, "movie"] 
            hllMovies.add(str(mov))
            #Users
            usr = dfStrPrWin.loc[m, "user"] 
            hllUsers.add(str(usr))
        winUniqueRcdC.loc[winN] = [len(hllMovies), len(hllUsers)]
        
        win = win + 1    
        start = start + strmStep
        end = end  + 1000 
        n =n + strmStep
    print("Bytes for Unique Movies HyperLogLog: ",  hllMovies.__sizeof__())
    print("Bytes for Unique Users HyperLogLog: ",  hllUsers.__sizeof__()) 
    #Evaluation
    if BQ2_1 == 1:
        rmseMov = mean_squared_error(winUniqueRcdB['unqMovies'], winUniqueRcdC['unqMovies'])
        rmseUsr = mean_squared_error(winUniqueRcdB['unqUsers'], winUniqueRcdC['unqUsers'])
        print("RMSE between Actual and Hyperloglog for Unique Movies: ",  rmseMov)
        print("RMSE between Actual and Hyperloglog for Unique Users: ",  rmseUsr)
        df_confusionMov = ConfusionMatrix(winUniqueRcdB['unqMovies'], winUniqueRcdC['unqMovies'])
        df_confusionUsr = ConfusionMatrix(winUniqueRcdB['unqUsers'], winUniqueRcdC['unqUsers'])
        print("Confusion Matrix for Movies: ")
        df_confusionMov.print_stats()
        print("Confusion Matrix for Users: ")
        df_confusionUsr.print_stats()
        
#Q3: Streaming processing
if BQ3 == 1:  
    sampleSize = sampleSize
    start =0
    end = 1000
    strmStep = 1000
    movsFrq = pd.DataFrame( data={'Frq': []}, index=None, columns=['Frq'])
    movsFrq.index.name='Movie'
    usrsFrq = pd.DataFrame( data={'Frq': []}, index=None, columns=['Frq'])
    usrsFrq.index.name='User'
    samplePriorMv = pd.DataFrame( data={ 'Frq':[]}, index=None, columns=[ 'Frq'])
    samplePriorMv.index.name='movie'
    samplePriorUs = pd.DataFrame( data={  'Frq':[]}, index=None, columns=[ 'Frq'])
    samplePriorUs.index.name='user'
    sampleResvrMv = pd.DataFrame( data={ 'Frq':[]}, index=None, columns=[ 'Frq'])
    sampleResvrMv.index.name='movie'
    sampleResvrUs = pd.DataFrame( data={ 'Frq':[]}, index=None, columns=[ 'Frq'])
    sampleResvrUs.index.name='user'
    win =0 
    n = 0
    
    tags=np.zeros(sampleSize)
    np.random.seed(1)
    while n < len(dfStrPr):
        i = 0 
        winN = 'Window_' + str(win) 
        dfStrPrWin = dfStrPr[start: end]
        for m , row in dfStrPrWin.iterrows():
            mov = dfStrPrWin.loc[m, "movie"] 
            usr = dfStrPrWin.loc[m, "user"]
         #Population
            if mov in movsFrq.index:
                movsFrq.loc[mov]['Frq'] = movsFrq.loc[mov]['Frq'] + 1
            else:
                movsFrq.loc[mov] = [1]
            if usr in samplePriorUs.index:
                usrsFrq.loc[usr]['Frq'] = usrsFrq.loc[usr]['Frq'] + 1
            else:
                usrsFrq.loc[usr] = [1]  
        #Sample    
            if i < sampleSize:
            #Priority Sampling
                if mov in samplePriorMv.index:
                    samplePriorMv.loc[mov]['Frq'] = samplePriorMv.loc[mov]['Frq'] + 1
                else:
                    samplePriorMv.loc[mov] = [1]
                if usr in samplePriorUs.index:
                    samplePriorUs.loc[usr]['Frq'] = samplePriorUs.loc[usr]['Frq'] + 1
                else:
                    samplePriorUs.loc[usr] = [1]  
                tags[i] = np.random.random()
                
            #Reservoir Sampling 
                if mov in sampleResvrMv.index:
                    sampleResvrMv.loc[mov]['Frq'] = sampleResvrMv.loc[mov]['Frq'] + 1
                else:
                    sampleResvrMv.loc[mov] = [1]
                if usr in sampleResvrUs.index:
                    sampleResvrUs.loc[usr]['Frq'] = sampleResvrUs.loc[usr]['Frq'] + 1
                else:
                    sampleResvrUs.loc[usr] = [1]
            else:
            #Priority Sampling
                newTag = np.random.random()
                maxTag = np.max(tags)
                idxMaxTag = np.argmax(tags) 
                if maxTag > newTag:
                    if mov in samplePriorMv.index:
                         samplePriorMv.loc[mov]['Frq'] = samplePriorMv.loc[mov]['Frq'] + 1
                    else:
                        samplePriorMv.loc[mov] = [1]
                    if usr in samplePriorUs.index:
                        samplePriorUs.loc[usr]['Frq'] = samplePriorUs.loc[usr]['Frq'] + 1
                    else:
                        samplePriorUs.loc[usr] = [1]  
                    tags[idxMaxTag]=newTag  
            
            #Reservoir Sampling 
                if np.random.random() <= sampleSize/float(i): 
                    dropMov = sampleResvrMv.sample(n=1).index.values.tolist()[0]
                    dropUsr = sampleResvrUs.sample(n=1).index.values.tolist()[0]
                    #discards randomply an existing item
                    sampleResvrMv.drop(dropMov)
                    sampleResvrUs.drop(dropUsr)
                    if mov in sampleResvrMv.index:
                        sampleResvrMv.loc[mov]['Frq'] = sampleResvrMv.loc[mov]['Frq'] + 1
                    else:
                        sampleResvrMv.loc[mov] = [1]
                    if usr in sampleResvrUs.index:
                        sampleResvrUs.loc[usr]['Frq'] = sampleResvrUs.loc[usr]['Frq'] + 1
                    else:
                        sampleResvrUs.loc[usr] = [1]
                    
            i = i + 1
        win = win + 1    
        start = start + strmStep
        end = end  + 1000 
        n =n + strmStep
    usrsFrq = usrsFrq.sort_values('Frq', ascending=False)
    t10Usrs = usrsFrq.head(10)
    movsFrq = movsFrq.sort_values('Frq', ascending=False)
    t10Movs = movsFrq.head(10)
    print("Top10 Users of Population: ")
    print(t10Usrs)
    print("Top10 Movies of Population: ")
    print(t10Movs)
    samplePriorUs = samplePriorUs.sort_values('Frq', ascending=False)
    samplePriorT10Usrs = samplePriorUs.head(10)
    samplePriorMv = samplePriorMv.sort_values('Frq', ascending=False)
    samplePriorT10Movs = samplePriorMv.head(10)
    print("Top10 Users by Priority Sampling: ")
    print(samplePriorT10Usrs)
    print("Top10 Movies by Priority Sampling: ")
    print(samplePriorT10Movs)
    sampleResvrUs = sampleResvrUs.sort_values('Frq', ascending=False)
    sampleResvrT10Usrs = sampleResvrUs.head(10)
    sampleResvrMv = sampleResvrMv.sort_values('Frq', ascending=False)
    sampleResvrT10Movs = sampleResvrMv.head(10) 
    print("Top10 Users by Reservoir Sampling: ")
    print(sampleResvrT10Usrs)
    print("Top10 Movies by Reservoir Sampling: ")
    print( sampleResvrT10Movs)
     
     
    rmse = mean_squared_error(t10Usrs, samplePriorT10Usrs)
    print("Users by Priority RMSE: ",  rmse)
     
    rmse = mean_squared_error(t10Movs, samplePriorT10Movs)
    print("Movies by Priority RMSE: ",  rmse)
     
    rmse = mean_squared_error(t10Usrs, sampleResvrT10Usrs)
    print("Users by Reservoir RMSE: ",  rmse)
   
    rmse = mean_squared_error(t10Movs, sampleResvrT10Movs)
    print("Movies by Reservoir RMSE: ",  rmse)