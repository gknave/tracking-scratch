import numpy as np
import pandas as pd

search_radius = 100

data = pd.load_csv('sh2.MP4.csv')

num_frames = np.max(data['Frame'].values)+1

track_counter = 0
current_tracks = {}
completed_tracks = {}

for k, row in data[data['Frame']==0].iterrows():
    key = 'track{:05d}'.format(track_counter)
    track_counter += 1
    current_tracks[key] = [(0, (row['Column'], row['Row'])),]

for time_step in range(1,num_frames):
    num_active = len(current_tracks.keys()) 
    now = np.zeros(num_active, 2)
    prior = np.zeros(num_active, 2)
    key_list = current_tracks.keys()
    for index, key in enumerate(key_list):
        now[index,:] = current_tracks[key][-1][1]
        if len(current_tracks[key])>1:
            prior[index,:] = current_tracks[key][-2][1]
        else:
            prior[index,:] = now[index,:]
    velocity = now-prior
    estimate = now+velocity

    current_agents = np.zeros(np.sum(data['Frame']==time_step), 2)
    for k, row in data[data['Frame']==time_step].iterrows():
        current_agents[k,:] = [row['Column'], row['Row']]
    costs = np.zeros(np.sum(data['Frame']==time_step),1) 
    links = np.zeros(np.sum(data['Frame']==time_step),1) 
    for index, key in enumerate(key_list):
        dist_list = (current_agents[0]-current_tracks[key][-1][1][0])^2+(current_agents[1]-current_tracks[key][-1][1][1])^2
        costs[index] = np.min(dist_list) 
        if costs[index] < search_radius**2:
            links[index] = np.argmin(dist_list)
