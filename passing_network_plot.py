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
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from pandas.io.json import json_normalize

## Import Selenium for scrape
from selenium import webdriver
import time

# Import ipywidgets
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets

def pitch():
    """
    code to plot a soccer pitch 
    """
    #create figure
    fig,ax=plt.subplots(1,1,figsize=(16,8))
    
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
    
    return ax

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
    
    global df,players_df,team_dict,team_df ,players_team_df
    
    driver = webdriver.Chrome(r"C:\Users\zhiyu\Dropbox\Yuan\Learning\chromedriver.exe")
 
    #match_id  = '1485276'    
# enter keyword to search 
    url = 'https://www.whoscored.com/Matches/%s/Live/'%match_id
    
    # Get to the website
    driver.get(url) 
    
    # get elements 
    matchCentredata = driver.execute_script("return matchCentreData")
    
    #elements = driver.find_elements_by_xpath("//div[@class='header-main__wrapper']") 
  
    # Return the matchCentredata as a dataframe

    df = pd.DataFrame(matchCentredata['events'])
    
    # Return the players_df dataframe
    players_df = json_normalize(matchCentredata["playerIdNameDictionary"])
    players_df = players_df.transpose()
    players_df['playerId'] = players_df.index
    players_df.columns = ['PlayerName', 'playerId'] 
    
    # Getting the teams dictionary
    away_df = pd.DataFrame({k:[v] for k,v in matchCentredata['away'].items() if k in ('name','teamId','field')})
    team_df = away_df.append(pd.DataFrame({k:[v] for k,v in matchCentredata['home'].items() if k in ('name','teamId','field')}))
    #team_df = team_df.set_index('teamId')
    team_df = team_df.rename(columns={'teamId': 'team_id'})
    team_dict = team_df['name'].to_dict()
    
    away_data = pd.DataFrame(json_normalize(matchCentredata["away"]))
    away_player_data = pd.DataFrame([flatten_json(x) for x in away_data['players'][0]])
    home_data = pd.DataFrame(json_normalize(matchCentredata["home"]))
    home_player_data = pd.DataFrame([flatten_json(x) for x in home_data['players'][0]])
    all_player_data = away_player_data[['field','playerId']].append(home_player_data[['field','playerId']])
    players_team_df = team_df.merge(all_player_data[['field','playerId']], how = 'left')
    
    return df

def prepare_joined_data():
    players_df['playerId'] = players_df.playerId.astype(str).astype(int)
    players_df['PlayerName'] = players_df.PlayerName.astype(str)
    test_2_df = test_df.merge(players_df,how = 'left', left_on = "playerId", right_on = 'playerId',suffixes=('_left', '_right'))
    test_3_df = test_2_df.merge(players_df,how = 'left', left_on = "Next_playerId", right_on = 'playerId',suffixes=('', '_next')).drop('playerId_next',axis=1)

    ## Joining the player pairs together
    test_3_df['PlayerName']= test_3_df['PlayerName'].fillna('-')
    test_3_df['PlayerName_next']= test_3_df['PlayerName_next'].fillna('-')
    test_3_df['ConcatName'] = test_3_df.apply(lambda x:'_'.join(sorted([x['PlayerName'],x['PlayerName_next']])),axis=1)
    
    return test_3_df

def compute_minutes(test_3_df,team_id):
    # Getting the card values
    Cards_df = test_3_df.query('teamId == %s'%team_id).cardType.dropna().apply(pd.Series)
    # Min for red/Yellow
    Cards_df = Cards_df.rename(columns={'displayName':'CardName'})
    test_3_df = test_3_df.join(Cards_df['CardName'])
    min_red = test_3_df[test_3_df.CardName.isin(["Red","SecondYellow"])]['minute'].min()
    # Min for Sub
    min_sub = test_3_df[test_3_df['type_displayName'] == 'SubstitutionOff']['minute'].min()
    # Time of match
    Ft_min = test_3_df[test_3_df['text'] == 'Second half ends']['minute'].min()
    
    #Get the min of everything
    max_min = np.nanmin([min_red,min_sub,Ft_min])
    
    return max_min

def prepare_passes_data(team_id):
    global pair_pass_df, df_passes, max_passes, player_coord, max_player_pass,player_total_pass
    
    #Return only valid passes for 11 players
    df_passes = test_3_df[(test_3_df['type_displayName'] =='Pass') & (test_3_df['minute'] < max_min)]
    #Use Median to get the median coordinates
    player_coord = df_passes.query("outcomeType_displayName == 'Successful' & teamId == %s"%team_id).groupby('PlayerName').agg({'x':'median','y':'median'})
    #Total passes per player
    df_total_pass = df_passes.groupby("PlayerName").size().to_frame()
    df_passes = df_passes.query("PlayerName != '-' & PlayerName_next != '-' ")
    #Getting the pair successful passes 
    pair_pass_df = df_passes.query("teamId == %s & type_displayName =='Pass' & outcomeType_value == 1"%team_id).groupby(['ConcatName']).size().to_frame().rename(columns={0:'Number of Passes'})
    
    #Getting max passes to for scaling 
    max_passes = pair_pass_df["Number of Passes"].max()
    
    #Creating a dataframe for total player passes and returning the max passes in the team for a player
    player_total_pass = player_coord.join(df_total_pass,how = 'left')
    player_total_pass = player_total_pass.rename(columns = {0:'Total_Player_Passes'})
    max_player_pass = player_total_pass['Total_Player_Passes'].max()
    

def plot_passing_network():
    ax = pitch()

    for key, col in pair_pass_df.iterrows():
        Player1, Player2 = key.split('_')
        pair_pass_df.at[key,'Player1'] = Player1
        pair_pass_df.at[key,'Player2'] = Player2
        Player1_x = player_coord.loc[Player1]['x']
        Player1_y = player_coord.loc[Player1]['y']
        Player2_x = player_coord.loc[Player2]['x']
        Player2_y = player_coord.loc[Player2]['y']

        line_width = _convert_range_passes(col['Number of Passes'],max_passes,10)

        #pair_pass_df.at[key,'linewidth'] = line_width
        #print (Player1,Player2,'r-o',line_width)

        ax.plot([Player1_x,Player2_x],[Player1_y,Player2_y],'r-',
                linestyle = '-',lw=line_width, alpha=1,zorder=3, markersize = 15,markerfacecolor = "White")

    ### Plot the playername as well as the size of the marker depending on number of passes    
    for PlayerName, txt in player_total_pass.iterrows():
        #print(PlayerName, txt['x'], txt['y'])
        marker_size = _convert_range_passes(txt['Total_Player_Passes'],max_player_pass,50)
        ax.plot(txt['x'], txt['y'],'ro',markersize = marker_size)
        ax.plot(txt['x'], txt['y'],'wo',markersize = marker_size-20,zorder=4)
        ax.annotate(PlayerName, xy =(txt['x']+1, txt['y']+1))
        
    return ax
        #plt.annotate(playername,xy = (x_coord, y_coord),xytext = (x_coord+1,y_coord+1))


if __name__ == '__main__ ':
    df = load_json2(match_id)
    test_df = cross_map_from_api(df)
    max_min = compute_minutes(test_3_df,team_id)
    prepare_passes_data(team_id)
    plot_passing_network()