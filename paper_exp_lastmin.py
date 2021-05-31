# -*- coding: utf-8 -*-
"""
Created on Fri May 28 21:07:00 2021

@author: chait
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import *
from rrt_paths import *
from sd_metric import *
from heatmap import *
import copy
import matplotlib.animation as animation 
import cv2
from multiprocessing import Pool
import time
import random
from scipy.stats import poisson

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Load image of ep6 plan %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# NOTE: 0.26m is one pixel
room_path = 'ep6_outline_smallest.jpg'
ep6_plan = plt.imread(room_path)
ep6_plan_copy = copy.deepcopy(ep6_plan)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get all user inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#### Get path to store results in ####
allPaths = np.load('results/metricTests/Exp3/num_ppl/ppl30/path.npy', allow_pickle=True)

results_path = 'results/metricTests/Exp3/num_ppl/ppl30/'
removed_pixels = None

air_change_percent = 50/100
air_change_rate = 10*60
hl = 9*60

N=30
kappa = 0.7*np.ones(N) #immune particles
v = 1*np.ones(N) # all not immune
#v[random.sample(range(0,N), int(100*30/100))]=0
rho_range = 68*np.ones(N) # age
zeta = 0.7*np.ones(N) # susceptibility
beta = 0.4*np.ones(N) # mask eff
eta = 0.7*np.ones(N) # infectiousness
mu=0.5*np.ones(N)

# Age groups:              0-4, 5-17, 18-29, 30-39, 40-49, 50-64, 65-74, 75-84, 85+
# Risk of hospitalization: 2x,  1x,   6x,    10x,   15x,   25x,   40x,   65x,   95x
# Ref: https://www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-age.html
rho = np.zeros(N)
for i in range(len(rho_range)):
    if rho_range[i] >= 0 and rho_range[i] <= 4:
        rho[i] = 2/259
    elif rho_range[i] > 4 and rho_range[i] <= 17:
        rho[i] = 1/259
    elif rho_range[i] > 17 and rho_range[i] <= 29:
        rho[i] = 6/259
    elif rho_range[i] > 29 and rho_range[i] <= 39:
        rho[i] = 10/259
    elif rho_range[i] > 39 and rho_range[i] <= 49:
        rho[i] = 15/259
    elif rho_range[i] > 49 and rho_range[i] <= 64:
        rho[i] = 25/259
    elif rho_range[i] > 64 and rho_range[i] <= 74:
        rho[i] = 40/259
    elif rho_range[i] > 74 and rho_range[i] <= 84:
        rho[i] = 65/259
    elif rho_range[i] > 84:
        rho[i] = 95/259
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Convert image to black and white and get boundaries and obstacles %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#         
r = ep6_plan[:,:,2]
bw_img = np.zeros_like(r)
for x in range(np.shape(r)[0]):
    for y in range(np.shape(r)[1]):
        mid = (255+1)/2
        if r[x,y] < mid + mid*0.5:
            bw_img[x,y] = 0
        else:
            bw_img[x,y] = 1      
bw_img = cv2.cvtColor(ep6_plan, cv2.COLOR_BGR2GRAY)
(thresh, bw_img) = cv2.threshold(bw_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
bw_img = cv2.threshold(bw_img, thresh, 255, cv2.THRESH_BINARY)[1]
indices = np.where(bw_img != [255])
obstacles = zip(indices[0], indices[1])
obstacles = list(set(obstacles))
################################################################################################################################
maxLen = 0
for n in range(N):
    if len(allPaths[n]) > maxLen:
        maxLen = len(allPaths[n])
        
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values, framewise_positions, all_distances = get_metric_values(allPaths, maxLen, alpha)

# Get SD metric for each person
sd_personwise = []
sd_avg = []
for i in range(len(framewise_positions)):
    sd_temp = []
    for n in range(N):
        sd = sd_metric_per_person(framewise_positions[i], n, alpha)
        sd_temp.append(sd)
    sd_personwise.append(sd_temp)
    sd_avg.append(np.mean(sd_temp))
    
sd_avg30=sd_avg
############################################################################################################################
    
R_reza, C_reza, pixels_to_color, final_inf = risk_detailed_reza_original(framewise_positions, kappa=kappa, v=v, rho=rho, 
                                    zeta=zeta, beta=beta, eta=eta, mu=1, D=0.003, air_change_percent=air_change_percent, 
                                    air_change_rate=air_change_rate, neighbor_dist=1,  F_s=1, hl=hl, 
                                    num_initial_inf=[5,2,10,16,27,23],
                                    ep6_plan=ep6_plan, XMIN1=12, XMAX1=115, YMIN1=24, YMAX1=210)

# Only for num_ppl
R_reza, C_reza, pixels_to_color, final_inf = risk_detailed_reza_original(framewise_positions, kappa=kappa, v=v, rho=rho, 
                                    zeta=zeta, beta=beta, eta=eta, mu=1, D=0.003, air_change_percent=air_change_percent, 
                                    air_change_rate=air_change_rate, neighbor_dist=1,  F_s=1, hl=hl, 
                                    num_initial_inf=[2,14,6],
                                    ep6_plan=ep6_plan, XMIN1=12, XMAX1=115, YMIN1=24, YMAX1=210)


r_path = results_path + 'R.npy'
np.save(r_path, R_reza)
c_path = results_path + 'C.npy'
np.save(c_path, C_reza)


R = []
R_avg = []
for i in range(len(R_reza)):
    R.append(R_reza[i])
    R_avg.append(np.mean(R_reza[i]))

x = np.arange(1,len(R_reza)+1, 5)
xlabel = []
for i in range(len(sd_personwise)):
    if i%5 == 0:
        xlabel.append(i+1)
R_avg30=R_avg[0:len(R_avg5)]
#sd_avg5=sd_avg

# Only for num_ppl
x = np.arange(1,len(R_avg5)+1, 5)
xlabel = []
for i in range(len(sd_personwise)):
    if i%5 == 0:
        xlabel.append(i+1)
R_avg30=R_avg[0:len(R_avg5)]


plt.figure(figsize=(15,10))
plt.plot(R_avg5, label='5 people')
plt.plot(R_avg15, 'o-', label='15 people')
plt.plot(R_avg30, '*-', label='30 people')
plt.xticks(x, xlabel, size=30)
plt.yticks(size=30)
plt.xlabel('Frame number', fontsize=30)
plt.ylabel('Average Risk Metric', fontsize=30)
plt.legend(prop={'size': 30})
plt.savefig('results/metricTests/Exp3/num_ppl/avgplots_ppl_combined.png')


# Only for num_ppl
plt.figure(figsize=(15,10))
plt.plot(np.array(sd_avg5), label='5 people')
plt.plot(np.array(sd_avg15[0:len(sd_avg5)]), 'o-', label='15 people')
plt.plot(np.array(sd_avg30[0:len(sd_avg5)]), '*-', label='30 people')
plt.xticks(x, xlabel, size=30)
plt.yticks(size=30)
plt.xlabel('Frame number', fontsize=30)
plt.ylabel('Average Social Distancing Metric', fontsize=30)
plt.legend(prop={'size': 30})
plt.savefig('results/metricTests/Exp3/num_ppl/avgsd_ppl_combined.png')



# Only for num_ppl
plt.figure(figsize=(15,10))
plt.plot(R_avg)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
plt.xlabel('Frame number', fontsize=18)
plt.ylabel('Average Risk Metric', fontsize=18)
plt.title('Number of people=30', fontsize=18)
plt.savefig('results/metricTests/Exp3/num_ppl/avgrisk_ppl30.png')

# Only for num_ppl
plt.figure(figsize=(15,10))
plt.plot(sd_avg)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
plt.xlabel('Frame number', fontsize=18)
plt.ylabel('Average Social Distancing Metric', fontsize=18)
plt.title('Number of people=5', fontsize=18)
plt.savefig('results/metricTests/Exp3/num_ppl/avgsd_ppl5.png')

##########################################################################################################################
####################################### EXPERIMENT 1########################################################
######################################################################################################################
# N = 2, 4, 10, 30
# Walking at speed of 1m per second or 1m per frame
results_path = 'results/metricTests/Exp1/'
outer_radius = 8/0.26
inner_radius = 1/0.26
distance = int(outer_radius - inner_radius)

# Define background image
background = 255*np.ones((100,100,3)).astype(np.uint8)

# Defining middle of space as (50,50)
mid = (50,50)

N = 2
start = []
end = []
positions2 = []
frames = []
d_in_m = [] # Distance from center in meters
for d in range(0, distance):
    temp_pos = []
    bg = deepcopy(background)
    r = outer_radius - d
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(r*np.cos(theta) + mid[0])
        y = int(r*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions2.append(temp_pos)
    frames.append(bg)
    d_in_m.append(str(round(r*0.26,1)))
    
for d in range(0, distance):
    temp_pos = []
    bg = deepcopy(background)
    r = inner_radius + d
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(r*np.cos(theta) + mid[0])
        y = int(r*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions2.append(temp_pos)
    frames.append(bg)
    d_in_m.append(str(round(r*0.26,1)))

N=4  
start = []
end = []
positions4 = []
frames = []
d_in_m = [] # Distance from center in meters
for d in range(0, distance):
    temp_pos = []
    bg = deepcopy(background)
    r = outer_radius - d
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(r*np.cos(theta) + mid[0])
        y = int(r*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions4.append(temp_pos)
    frames.append(bg)
    d_in_m.append(str(round(r*0.26,1)))
    
for d in range(0, distance):
    temp_pos = []
    bg = deepcopy(background)
    r = inner_radius + d
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(r*np.cos(theta) + mid[0])
        y = int(r*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions4.append(temp_pos)
    frames.append(bg)
    d_in_m.append(str(round(r*0.26,1)))

N=10    
start = []
end = []
positions10 = []
frames = []
d_in_m = [] # Distance from center in meters
for d in range(0, distance):
    temp_pos = []
    bg = deepcopy(background)
    r = outer_radius - d
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(r*np.cos(theta) + mid[0])
        y = int(r*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions10.append(temp_pos)
    frames.append(bg)
    d_in_m.append(str(round(r*0.26,1)))
    
for d in range(0, distance):
    temp_pos = []
    bg = deepcopy(background)
    r = inner_radius + d
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(r*np.cos(theta) + mid[0])
        y = int(r*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions10.append(temp_pos)
    frames.append(bg)
    d_in_m.append(str(round(r*0.26,1)))

N=30    
start = []
end = []
positions30 = []
frames = []
d_in_m = [] # Distance from center in meters
for d in range(0, distance):
    temp_pos = []
    bg = deepcopy(background)
    r = outer_radius - d
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(r*np.cos(theta) + mid[0])
        y = int(r*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions30.append(temp_pos)
    frames.append(bg)
    d_in_m.append(str(round(r*0.26,1)))
    
for d in range(0, distance):
    temp_pos = []
    bg = deepcopy(background)
    r = inner_radius + d
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(r*np.cos(theta) + mid[0])
        y = int(r*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions30.append(temp_pos)
    frames.append(bg)
    d_in_m.append(str(round(r*0.26,1)))
    
    

allPaths2 = []
for n in range(2):
    temp_path = []
    for i in range(len(positions2)):
        temp_path.append(positions2[i][n])
    allPaths2.append(temp_path)
    
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values2, framewise_positions2, all_distances2 = get_metric_values(allPaths2, len(allPaths2[0]), alpha)
        
allPaths4 = []
for n in range(4):
    temp_path = []
    for i in range(len(positions4)):
        temp_path.append(positions4[i][n])
    allPaths4.append(temp_path)
    
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values4, framewise_positions4, all_distances4 = get_metric_values(allPaths4, len(allPaths4[0]), alpha)
        
allPaths10 = []
for n in range(10):
    temp_path = []
    for i in range(len(positions10)):
        temp_path.append(positions10[i][n])
    allPaths10.append(temp_path)
    
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values10, framewise_positions10, all_distances10 = get_metric_values(allPaths10, len(allPaths10[0]), alpha)
        
allPaths30 = []
for n in range(30):
    temp_path = []
    for i in range(len(positions30)):
        temp_path.append(positions30[i][n])
    allPaths30.append(temp_path)
    
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values30, framewise_positions30, all_distances30 = get_metric_values(allPaths30, len(allPaths30[0]), alpha)
        


N=2
R_reza2, C_reza2, pixels_to_color, _ = risk_detailed_reza_original(framewise_positions2, kappa=0.7*np.ones(N), v=1*np.ones(N), rho=0.2*np.ones(N), 
                                    zeta=0.7*np.ones(N), beta=0.3*np.ones(N), eta=0.7*np.ones(N), mu=1, D=0.003, 
                                    air_change_percent=0.5, air_change_rate=10*60, neighbor_dist=1,  F_s=1, hl=9*60,num_initial_inf=2,
                                    ep6_plan=background, XMIN1=0, XMAX1=100, YMIN1=0, YMAX1=100)
R_avg2 = []
for i in range(len(R_reza2)):
    R_avg2.append(np.mean(R_reza2[i]))

N=4
R_reza4, C_reza4, pixels_to_color, _ = risk_detailed_reza_original(framewise_positions4, kappa=0.7*np.ones(N), v=1*np.ones(N), rho=0.2*np.ones(N), 
                                    zeta=0.7*np.ones(N), beta=0.3*np.ones(N), eta=0.7*np.ones(N), mu=1, D=0.003, 
                                    air_change_percent=0.5, air_change_rate=10*60, neighbor_dist=1,  F_s=1, hl=9*60,num_initial_inf=4,
                                    ep6_plan=background, XMIN1=0, XMAX1=100, YMIN1=0, YMAX1=100)
R_avg4 = []
for i in range(len(R_reza4)):
    R_avg4.append(np.mean(R_reza4[i]))

N=10
R_reza10, C_reza10, pixels_to_color, _ = risk_detailed_reza_original(framewise_positions10, kappa=0.7*np.ones(N), v=1*np.ones(N), rho=0.2*np.ones(N), 
                                    zeta=0.7*np.ones(N), beta=0.3*np.ones(N), eta=0.7*np.ones(N), mu=1, D=0.003, 
                                    air_change_percent=0.5, air_change_rate=10*60, neighbor_dist=1,  F_s=1, hl=9*60,num_initial_inf=10,
                                    ep6_plan=background, XMIN1=0, XMAX1=100, YMIN1=0, YMAX1=100)
R_avg10 = []
for i in range(len(R_reza10)):
    R_avg10.append(np.mean(R_reza10[i]))

N=30
R_reza30, C_reza30, pixels_to_color, _ = risk_detailed_reza_original(framewise_positions30, kappa=0.7*np.ones(N), v=1*np.ones(N), rho=0.2*np.ones(N), 
                                    zeta=0.7*np.ones(N), beta=0.3*np.ones(N), eta=0.7*np.ones(N), mu=1, D=0.003, 
                                    air_change_percent=0.5, air_change_rate=10*60, neighbor_dist=1,  F_s=1, hl=9*60,num_initial_inf=30,
                                    ep6_plan=background, XMIN1=0, XMAX1=100, YMIN1=0, YMAX1=100)
R_avg30 = []
for i in range(len(R_reza30)):
    R_avg30.append(np.mean(R_reza30[i]))



x = np.arange(1,len(sd_metric_values30)+1, 5)
xlabel = []
for i in range(len(sd_metric_values30)):
    if i%5 == 0:
        xlabel.append(d_in_m[i])
        
plt.figure(figsize=(15,10))
#plt.plot(sd_metric_values2, label='2 people')
plt.plot(sd_metric_values30/(30*(33)), label='30 people')
plt.plot(sd_metric_values10/(10*(10)), 'o-', label='10 people')
plt.plot(sd_metric_values4/(4*(4)), '*-', label='4 people')
plt.xticks(x, xlabel, size=30)
plt.yticks(size=30)
plt.xlabel('Average distance between people in meters', fontsize=30)
plt.ylabel('Average Social Distancing Metric', fontsize=30)
plt.legend(prop={'size': 30})
plt.savefig('results/metricTests/Exp1/sdmetric_exp1_avg.png')


       
plt.figure(figsize=(15,10))
plt.plot(R_avg10, label='30 people')
plt.plot(R_avg4, 'o-', label='10 people')
plt.plot(R_avg30, '*-', label='4 people')
plt.xticks(x, xlabel, size=30)
plt.yticks(size=30)
plt.xlabel('Average distance between people in meters', fontsize=30)
plt.ylabel('Average Risk Metric', fontsize=30)
plt.legend(prop={'size': 30})
plt.savefig('results/metricTests/Exp1/riskmetric_exp1_avg.png')



##########################################################################################################################
####################################### EXPERIMENT 2 ########################################################
######################################################################################################################
results_path = 'results/metricTests/Exp2/'
N = 5
T = 60
# Define background image
background = 255*np.ones((100,100,3)).astype(np.uint8)

# Defining middle of space as (50,50)
mid = (50,50)

radius = 1/0.26
# Position people on circle
start = []
end = []
positions1 = []
frames = []
for t in range(T):
    temp_pos = []
    bg = deepcopy(background)
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(radius*np.cos(theta) + mid[0])
        y = int(radius*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions1.append(temp_pos)
    frames.append(bg)

radius = 2/0.26
# Position people on circle
start = []
end = []
positions2 = []
frames = []
for t in range(T):
    temp_pos = []
    bg = deepcopy(background)
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(radius*np.cos(theta) + mid[0])
        y = int(radius*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions2.append(temp_pos)
    frames.append(bg)
    
radius = 6/0.26
# Position people on circle
start = []
end = []
positions6 = []
frames = []
for t in range(T):
    temp_pos = []
    bg = deepcopy(background)
    for n in range(N):
        theta = n*2*np.pi/N
        x = int(radius*np.cos(theta) + mid[0])
        y = int(radius*np.sin(theta) + mid[0])
        temp_pos.append((x,y))
        bg[int(x), int(y), :] = [0,0,0]
    positions6.append(temp_pos)
    frames.append(bg)


############################################################ Get SD metric ##################################################
# Convert positions to required format
allPaths1 = []
for n in range(N):
    temp_path = []
    for i in range(len(positions1)):
        temp_path.append(positions1[i][n])
    allPaths1.append(temp_path)
    
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values1, framewise_positions1, all_distances1 = get_metric_values(allPaths1, len(allPaths1[0]), alpha)

allPaths2 = []
for n in range(N):
    temp_path = []
    for i in range(len(positions2)):
        temp_path.append(positions2[i][n])
    allPaths2.append(temp_path)
    
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values2, framewise_positions2, all_distances2 = get_metric_values(allPaths2, len(allPaths2[0]), alpha)

allPaths6 = []
for n in range(N):
    temp_path = []
    for i in range(len(positions6)):
        temp_path.append(positions6[i][n])
    allPaths6.append(temp_path)
    
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values6, framewise_positions6, all_distances6 = get_metric_values(allPaths6, len(allPaths6[0]), alpha)



############################################################## Get risk metric ###############################################
R_reza1, C_reza, pixels_to_color, _ = risk_detailed_reza_original(framewise_positions1, kappa=0.7*np.ones(N), v=1*np.ones(N), rho=0.2*np.ones(N), 
                                    zeta=0.7*np.ones(N), beta=0.3*np.ones(N), eta=0.7*np.ones(N), mu=1, D=0.003, 
                                    air_change_percent=0.5, air_change_rate=10*60, neighbor_dist=1,  F_s=1, hl=9*60,num_initial_inf=3,
                                    ep6_plan=background, XMIN1=0, XMAX1=100, YMIN1=0, YMAX1=100)
R_avg1 = []
for i in range(len(R_reza1)):
    R_avg1.append(np.mean(R_reza1[i]))

R_reza2, C_reza, pixels_to_color, _ = risk_detailed_reza_original(framewise_positions2, kappa=0.7*np.ones(N), v=1*np.ones(N), rho=0.2*np.ones(N), 
                                    zeta=0.7*np.ones(N), beta=0.3*np.ones(N), eta=0.7*np.ones(N), mu=1, D=0.003, 
                                    air_change_percent=0.5, air_change_rate=10*60, neighbor_dist=1,  F_s=1, hl=9*60,num_initial_inf=3,
                                    ep6_plan=background, XMIN1=0, XMAX1=100, YMIN1=0, YMAX1=100)
R_avg2 = []
for i in range(len(R_reza2)):
    R_avg2.append(np.mean(R_reza2[i]))

R_reza6, C_reza, pixels_to_color, _ = risk_detailed_reza_original(framewise_positions6, kappa=0.7*np.ones(N), v=1*np.ones(N), rho=0.2*np.ones(N), 
                                    zeta=0.7*np.ones(N), beta=0.3*np.ones(N), eta=0.7*np.ones(N), mu=1, D=0.003, 
                                    air_change_percent=0.5, air_change_rate=10*60, neighbor_dist=1,  F_s=1, hl=9*60,num_initial_inf=3,
                                    ep6_plan=background, XMIN1=0, XMAX1=100, YMIN1=0, YMAX1=100)
R_avg6 = []
for i in range(len(R_reza6)):
    R_avg6.append(np.mean(R_reza6[i]))
    

xlabel = []
x = np.arange(1,len(R_avg6)+1, 5)
for i in range(len(R_avg6)):
    if i%5 == 0:
        xlabel.append(i)

plt.figure(figsize=(15,10))
plt.plot(R_avg1, label='1m radius circle')
plt.plot(R_avg2, 'o-', label='2m radius circle')
plt.plot(R_avg6, '*-', label='6m radius circle')
plt.xticks(x, xlabel, size=30)
plt.yticks(size=30)
plt.xlabel('Average distance between people in meters', fontsize=30)
plt.ylabel('Average Risk Metric', fontsize=30)
plt.legend(prop={'size': 30})
plt.savefig('results/metricTests/Exp2/riskmetric_exp2_avg.png')