#The basics
import pandas as pd
import numpy as np
import json
import seaborn as sns

#Plotting
import matplotlib.pyplot as plt


#Statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf 

from matplotlib.patches import Arc
from matplotlib.patches import Ellipse
from pandas.io.json import json_normalize

## Import Selenium for scrape
from selenium import webdriver
import time


# Import ipywidgets
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets

%matplotlib inline

## Plotting pitch

def pitch():
    """
    code to plot a soccer pitch 
    """
    #create figure
    
    fig,ax=plt.subplots(figsize=(7,5))
    
    #Pitch Outline & Centre Line
    plt.plot([0,0],[0,100], color="black")
    plt.plot([0,100],[100,100], color="black")
    plt.plot([100,100],[100,0], color="black")
    plt.plot([100,0],[0,0], color="black")
    plt.plot([50,50],[0,100], color="black")

    #Left Penalty Area
    plt.plot([16.5,16.5],[80,20],color="black")
    plt.plot([0,16.5],[80,80],color="black")
    plt.plot([16.5,0],[20,20],color="black")

    #Right Penalty Area
    plt.plot([83.5,100],[80,80],color="black")
    plt.plot([83.5,83.5],[80,20],color="black")
    plt.plot([83.5,100],[20,20],color="black")

    #Left 6-yard Box
    plt.plot([0,5.5],[65,65],color="black")
    plt.plot([5.5,5.5],[65,35],color="black")
    plt.plot([5.5,0.5],[35,35],color="black")

    #Right 6-yard Box
    plt.plot([100,94.5],[65,65],color="black")
    plt.plot([94.5,94.5],[65,35],color="black")
    plt.plot([94.5,100],[35,35],color="black")
    
        
    ## Goals
    ly4 = [46.34,46.34,53.66,53.66]
    lx4 = [100,100.2,100.2,100]
    plt.plot(lx4,ly4,color='black',zorder=5)

    ly5 = [46.34,46.34,53.66,53.66]
    lx5 = [0,-0.2,-0.2,0]
    plt.plot(lx5,ly5,color='black',zorder=5)

    #Prepare Circles
    centreCircle = Ellipse((50, 50), width=30, height=39, edgecolor="black", facecolor="None", lw=1.8)
    centreSpot = Ellipse((50, 50), width=1, height=1.5, edgecolor="black", facecolor="black", lw=1.8)
    leftPenSpot = Ellipse((11, 50), width=1, height=1.5, edgecolor="black", facecolor="black", lw=1.8)
    rightPenSpot = Ellipse((89, 50), width=1, height=1.5, edgecolor="black", facecolor="black", lw=1.8)

    #Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)
    
    #limit axis
    #plt.xlim(-1,101)
    #plt.ylim(0,100)
    
    ax.annotate("", xy=(25, 5), xytext=(5, 5),
                arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.text(7,7,'Attack',fontsize=20)
    
    return fig,ax

def flatten_json(nested_json, exclude=['']):
    """Flatten json object with nested keys into a single level.
        Args:
            nested_json: A nested json object.
            exclude: Keys to exclude from output.
        Returns:
            The flattened json object if successful, None otherwise.
    """
    out = {}

    def flatten(x, name='', exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude: flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out

def load_json2(match_id):
    
    global df,players_df 
    
    driver = webdriver.Chrome(r"C:\Users\zhiyu\Dropbox\Yuan\Learning\chromedriver.exe")
 
    #match_id  = '1485276'    
    # enter keyword to search 
    url = 'https://www.whoscored.com/Matches/%s/Live/'%match_id
    
    # Get to the website
    driver.get(url) 
    
    # get elements 
    matchCentredata = driver.execute_script("return matchCentreData")
    
    #elements = driver.find_elements_by_xpath("//div[@class='header-main__wrapper']") 
  
    
    # print complete elements list 
    #print(matchCentredata['events']) 

# Return the matchCentredata as a dataframe

    df = pd.DataFrame(matchCentredata['events'])
    
    # Return the players_df dataframe
    players_df = json_normalize(matchCentredata["playerIdNameDictionary"])
    players_df = players_df.transpose()
    players_df['playerId'] = players_df.index
    players_df.columns = ['PlayerName', 'playerId'] 
    
    return df



def cross_map_from_api(df):
      
    '''
    Can only be used with load_json2
    '''
    
    global new_df, test_df, flatten_qualifiers_df
    #df = load_json(jsonpath)
    
    #unpack outcome and type columns
    outcome_df = pd.concat([pd.DataFrame(json_normalize(x)) for x in df['outcomeType']],ignore_index=True)
    outcome_df.columns = ['outcomeType_displayName','outcomeType_value'] 

    type_df = pd.concat([pd.DataFrame(json_normalize(x)) for x in df['type']],ignore_index=True)
    #type_df.columns = ['idx','len','type_displayName','type_value'] 
    
    #Create new_df
    
    new_df = df.join(outcome_df.join(type_df[['displayName','value']]))
    
    ## Rename
    
    new_df = new_df.rename(columns={'displayName':'type_displayName',
                       'value':'type_value'})
    
    #Get next player
    new_df['Next_playerId'] = new_df['playerId'].shift(-1)
    
    #Create a dataframe for qualifiers to get crossing data
    flatten_qualifiers_df = pd.DataFrame([flatten_json(x) for x in new_df['qualifiers']])
    
    #Creating Crossing column
    flatten_qualifiers_df['isCross'] = 0

    for index, rows in flatten_qualifiers_df.iterrows():
    # Iterate over two given columns 
    # only from the dataframe 
        for column in list(flatten_qualifiers_df): 
            if rows[column] == 'Cross':
                flatten_qualifiers_df.at[index,'isCross'] = 1
                #print (index, rows[column])
    
    test_df = new_df.join(flatten_qualifiers_df['isCross'])

                
    return test_df


def plot_Cross_map(test_df,team_id):
  
    
    fig = pitch()

## Successful crosses

    plt.plot(test_df.query('teamId == %s & isCross == 1 & outcomeType_value == 1'%team_id)['x'],
             test_df.query('teamId == %s & isCross == 1 & outcomeType_value == 1'%team_id)['y'], 
             'g'+'o', alpha=0.5 )

    for i,row in test_df.query("teamId == %s & isCross == 1 & outcomeType_value == 1"%team_id).iterrows():
        plt.annotate("", xy=row[['endX','endY']], 
                 xytext=(row[['x','y']]),
                 alpha=0.5, arrowprops=dict(alpha=0.5,width=0.5,headlength=4.0,headwidth=4.0,color='g'),annotation_clip=False)

## Unsuccessful crosses

    plt.plot(test_df.query("teamId == %s & isCross == 1 & outcomeType_value == 0"%team_id)['x'],
             test_df.query("teamId == %s & isCross == 1 & outcomeType_value == 0"%team_id)['y'], 
             'r'+'o', alpha=0.5 )

    for i,row in test_df.query("teamId == %s & isCross == 1 & outcomeType_value == 0"%team_id).iterrows():
        plt.annotate("", xy=row[['endX','endY']], 
                 xytext=(row[['x','y']]),
                 alpha=0.5, arrowprops=dict(alpha=0.5,width=0.5,headlength=4.0,headwidth=4.0,color='r'),annotation_clip=False)


def plot_Pass_map(test_df,team_id,playerid = 0):
    
    fig = pitch()

## Successful crosses
    if playerid == 0:
        plt.plot(test_df.query("teamId == %s & type_displayName =='Pass' & outcomeType_value == 1"%(team_id))['x'],
                 test_df.query("teamId == %s & type_displayName =='Pass' & outcomeType_value == 1"%(team_id))['y'], 
                 'g'+'o', alpha=0.5 )

        for i,row in test_df.query("teamId == %s & type_displayName =='Pass' & outcomeType_value == 1"%(team_id)).iterrows():
            plt.annotate("", xy=row[['endX','endY']], 
            xytext=(row[['x','y']]),
            alpha=0.5, arrowprops=dict(alpha=0.5,width=0.5,headlength=4.0,headwidth=4.0,color='g'),annotation_clip=False)

## Unsuccessful crosses

        plt.plot(test_df.query("teamId == %s & type_displayName =='Pass' & outcomeType_value == 0"%(team_id))['x'],
                 test_df.query("teamId == %s & type_displayName =='Pass' & outcomeType_value == 0"%(team_id))['y'], 
                 'r'+'o', alpha=0.5 )

        for i,row in test_df.query("teamId == %s & type_displayName =='Pass' & outcomeType_value == 0"%(team_id)).iterrows():
            plt.annotate("", xy=row[['endX','endY']], 
                     xytext=(row[['x','y']]),
                     alpha=0.5, arrowprops=dict(alpha=0.5,width=0.5,headlength=4.0,headwidth=4.0,color='r'),annotation_clip=False)

    else:
        plt.plot(test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 1"%(team_id,playerid))['x'],
                   test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 1"%(team_id,playerid))['y'],
                 'g'+'o', alpha=0.5 )

        for i,row in test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 1"%(team_id,playerid)).iterrows():
            plt.annotate("", xy=row[['endX','endY']], 
            xytext=(row[['x','y']]),
            alpha=0.5, arrowprops=dict(alpha=0.5,width=0.5,headlength=4.0,headwidth=4.0,color='g'),annotation_clip=False)

## Unsuccessful crosses

        plt.plot(test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 0"%(team_id,playerid))['x'],
                  test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 0"%(team_id,playerid))['y'], 
                  'r'+'o', alpha=0.5 )
    
        for i,row in test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 0"%(team_id,playerid)).iterrows():
            plt.annotate("", xy=row[['endX','endY']], 
                     xytext=(row[['x','y']]),
                     alpha=0.5, arrowprops=dict(alpha=0.5,width=0.5,headlength=4.0,headwidth=4.0,color='r'),annotation_clip=False)



def plot_Pass_map2(test_df,team_id,playerid):
    
    fig = pitch()

## Successful crosses
  
    plt.plot(test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 1"%(team_id,playerid))['x'],
                   test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 1"%(team_id,playerid))['y'],
                 'g'+'o', alpha=0.5 )

    for i,row in test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 1"%(team_id,playerid)).iterrows():
            plt.annotate("", xy=row[['endX','endY']], 
            xytext=(row[['x','y']]),
            alpha=0.5, arrowprops=dict(alpha=0.5,width=0.5,headlength=4.0,headwidth=4.0,color='g'),annotation_clip=False)

## Unsuccessful crosses

    plt.plot(test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 0"%(team_id,playerid))['x'],
                  test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 0"%(team_id,playerid))['y'], 
                  'r'+'o', alpha=0.5 )
    
    for i,row in test_df.query("teamId == %s & playerId == %s & type_displayName =='Pass' & outcomeType_value == 0"%(team_id,playerid)).iterrows():
            plt.annotate("", xy=row[['endX','endY']], 
                     xytext=(row[['x','y']]),
                     alpha=0.5, arrowprops=dict(alpha=0.5,width=0.5,headlength=4.0,headwidth=4.0,color='r'),annotation_clip=False)


if __name__ == '__main__ ':
    df = load_json2(match_id)
    test_df = cross_map_from_api(df)
    plot_Pass_map(test_df,team_id)
    plot_Pass_map2(test_df,team_id,playerid)
    plot_Cross_map(test_df,team_id)
    