# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:34:54 2021

@author: chait
"""

import numpy as np
from copy import deepcopy
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from sd_metric import risk_detailed_reza_original, get_metric_values, sd_metric_per_person, diffusion
from utils import animate
from heatmap import dynamic_heatmap_model2

###############################################################################################################################
################################# Experiment 1 - N people walking towards and then away from each other #######################
###############################################################################################################################
# Simulate experiments to test metrics in a plane box with no obstacles and following defined paths

# N = 2, 4, 10, 30
# Walking at speed of 1m per second or 1m per frame
results_path = input('Enter path to store results (like results/exp1/)): ')
N = int(input('Enter number of people in simulation: '))
outer_radius = int(input('Enter diameter of starting circle in m (13m max): '))/0.26
inner_radius = int(input('Enter diameter of stopping circle in m: '))/0.26
distance = int(outer_radius - inner_radius)

# Define background image
background = 255*np.ones((100,100,3)).astype(np.uint8)

# Defining middle of space as (50,50)
mid = (50,50)

start = []
end = []
positions = []
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
    positions.append(temp_pos)
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
    positions.append(temp_pos)
    frames.append(bg)
    d_in_m.append(str(round(r*0.26,1)))
    
## Plot walking
fig = plt.figure()
display_overlay = []
for k in range(len(frames)):
    display_overlay.append([plt.imshow(frames[k],animated=True)])
ani = animation.ArtistAnimation(fig, display_overlay, interval=200, blit=True, repeat_delay=1000) # interval=300
plt.show()  
# Save animation
path = results_path + 'pplWalking_' + str(N) +  '.gif' 
ani.save(path, writer='imagemagick', fps=3)


################################################## Get SD metric ######################################################
# Convert positions to required format
allPaths = []
for n in range(N):
    temp_path = []
    for i in range(len(positions)):
        temp_path.append(positions[i][n])
    allPaths.append(temp_path)
    
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values, framewise_positions, all_distances = get_metric_values(allPaths, len(allPaths[0]), alpha)
        
# Get SD metric for each person
sd_personwise = []
for i in range(len(framewise_positions)):
    sd_temp = []
    for n in range(N):
        sd = sd_metric_per_person(framewise_positions[i], n, alpha)
        sd_temp.append(sd)
    sd_personwise.append(sd_temp)
    

# Box plot of SD metric for each individual
duration = 20 # in sec
refreshPeriod = 1000 # in ms
x = np.arange(1,len(sd_personwise)+1, 5)
xlabel = []
for i in range(len(sd_personwise)):
    if i%5 == 0:
        xlabel.append(d_in_m[i])
fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(sd_personwise, showfliers=False, medianprops=medianprops)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Average distance between people in meters')
ax.set_ylabel('SD metric value')
vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax.set_xlim(0,len(sd_personwise)+1)
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
ani = animation.FuncAnimation(fig, animate, fargs=(vl,refreshPeriod), blit= True, interval=200, repeat_delay=1000, repeat=True)
plt.show()
path = results_path + 'SDmetricBoxPlot_' + str(N) + '.gif' 
ani.save(path, writer='imagemagick', fps=3)


####################################################### Get risk metric #####################################################
R_reza, C_reza, pixels_to_color, _ = risk_detailed_reza_original(positions, kappa=0.7*np.ones(N), v=1*np.ones(N), rho=0.2*np.ones(N), 
                                    zeta=0.7*np.ones(N), beta=0.3*np.ones(N), eta=0.7*np.ones(N), mu=1, D=0.003, 
                                    air_change_percent=0.5, air_change_rate=10*60, neighbor_dist=1,  F_s=1, hl=9*60,num_initial_inf=23,
                                    ep6_plan=background, XMIN1=0, XMAX1=100, YMIN1=0, YMAX1=100)

# Save R and C since they take so long to run
R_path = results_path + 'R.npy'
np.save(R_path, R_reza)
C_path = results_path + 'C.npy'
np.save(C_path, C_reza)
    
# Load R and Cif necessary
R_reza = np.load(results_path+'R.npy')
C_reza = np.load(results_path+'C.npy')  
  
# Prepare R for plotting
R = []
for i in range(len(R_reza)):
    R.append(R_reza[i])
        
# Box plot for risk metric        
duration = 20 # in sec
refreshPeriod = 1000 # in ms
x = np.arange(1,len(R)+1,5)
xlabel = []
for i in range(len(R)):
    if i%5 == 0:
        xlabel.append(d_in_m[i])
fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(R, showfliers=False, medianprops=medianprops)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Average distance between people in meters')
ax.set_ylabel('Risk Metric')
vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax.set_xlim(0,len(R))
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
ani = animation.FuncAnimation(fig, animate, fargs=(vl,refreshPeriod), blit= True, interval=200, repeat_delay=1000, repeat=True)
plt.show()
path = results_path + 'RiskBoxPlot_' +  str(N) + '.gif' 
ani.save(path, writer='imagemagick', fps=3)

fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(R, showfliers=False, medianprops=medianprops)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Average distance between people in meters')
ax.set_ylabel('Risk Metric')
ax.set_xlim(0,len(R))
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
plt.show()
fig.set_size_inches((15, 7), forward=False)
path = results_path + 'riskmetric_exp1_n30.png'
plt.savefig(path)
########################################################### Generate heatmap ################################################
hm_reza, hm_ppl_reza = dynamic_heatmap_model2(C_reza, framewise_positions, background)

fig = plt.figure()
display_overlay = []
for k in range(len(hm_reza)):
    display_overlay.append([plt.imshow(hm_reza[k],animated=True)])
ani = animation.ArtistAnimation(fig, display_overlay, interval=200, blit=True, repeat_delay=1000) # interval=300
plt.show()

path = results_path + 'heatmap_' +  str(N) + '.gif' 
ani.save(path, writer='imagemagick', fps=3)




############################################################################################################################
##################### Experiment 2 - 5 people remain at distance D from center for time T. Vary T and D ####################
############################################################################################################################
results_path = input('Enter path to store results (like results/exp1/)): ')
N = int(input('Enter even number of people in simulation: '))
radius = int(input('Enter radius of circle to stand in in m (13m max): '))/0.26
T = int(input('Enter time for which you want people to stay at same position in seconds: '))

# Define background image
background = 255*np.ones((100,100,3)).astype(np.uint8)

# Defining middle of space as (50,50)
mid = (50,50)

# Position people on circle
start = []
end = []
positions = []
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
    positions.append(temp_pos)
    frames.append(bg)

    
## Plot walking
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(frames[0]) 
# Save image
path = results_path + 'pplWalking_' + str(N) +  '.png' 
fig.savefig(path)


############################################################ Get SD metric ##################################################
# Convert positions to required format
allPaths = []
for n in range(N):
    temp_path = []
    for i in range(len(positions)):
        temp_path.append(positions[i][n])
    allPaths.append(temp_path)
    
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values, framewise_positions, all_distances = get_metric_values(allPaths, len(allPaths[0]), alpha)

# Get SD metric for each person
sd_personwise = []
for i in range(len(framewise_positions)):
    sd_temp = []
    for n in range(N):
        sd = sd_metric_per_person(framewise_positions[i], n, alpha)
        sd_temp.append(sd)
    sd_personwise.append(sd_temp)
        
# Box plot for sd metric        
duration = 20 # in sec
refreshPeriod = 1000 # in ms
x = np.arange(1,len(sd_personwise)+1,5)
xlabel = []
for i in range(len(sd_personwise)):
    if i%5 == 0:
        xlabel.append(i)
fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(sd_personwise, showfliers=False, medianprops=medianprops)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Frame number')
ax.set_ylabel('Social distancing metric')
vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax.set_xlim(0,len(R))
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
ani = animation.FuncAnimation(fig, animate, fargs=(vl,refreshPeriod), blit= True, interval=200, repeat_delay=1000, repeat=True)
plt.show()
path = results_path + 'RiskBoxPlot_' +  str(N) + '.gif' 
ani.save(path, writer='imagemagick', fps=3)


############################################################## Get risk metric ###############################################
R_reza, C_reza, pixels_to_color, _ = risk_detailed_reza_original(positions, kappa=0.7*np.ones(N), v=1*np.ones(N), rho=0.2*np.ones(N), 
                                    zeta=0.7*np.ones(N), beta=0.3*np.ones(N), eta=0.7*np.ones(N), mu=1, D=0.003, 
                                    air_change_percent=0.5, air_change_rate=10*60, neighbor_dist=1,  F_s=1, hl=9*60,num_initial_inf=3,
                                    ep6_plan=background, XMIN1=0, XMAX1=100, YMIN1=0, YMAX1=100)

R = []
for i in range(len(R_reza)):
    R.append(R_reza[i])

fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(R, showfliers=False, medianprops=medianprops)
x = np.arange(1,len(R)+1,5)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Frame number')
ax.set_ylabel('Risk Metric')
ax.set_xlim(0,len(R))
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
plt.show()
fig.set_size_inches((15, 7), forward=False)
path = results_path + 'riskmetric_exp2_6m.png'
plt.savefig(path)

## Save numpy files
#r_path = results_path + 'R.npy'
#np.save(r_path, R_reza)
#c_path = results_path + 'C.npy'
#np.save(c_path, C_reza)
#
## Load R and Cif necessary
#R_reza = np.load(results_path+'R.npy')
#C_reza = np.load(results_path+'C.npy') 

# Box plot for risk metric
R = []
for i in range(len(R_reza)):
    R.append(R_reza[i])

# Select alternate values to plot for readability
r_plot = []
x_plot = []
for i in range(len(R)):
    if i%2 == 0:
        r_plot.append(R[i])
        x_plot.append(i+1)
        
# Box plot for risk metric        
duration = 20 # in sec
refreshPeriod = 1000 # in ms
x = np.arange(1,len(R)+1,5)
xlabel = []
for i in range(len(R)):
    if i%5 == 0:
        xlabel.append(i)
fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(R, showfliers=False, medianprops=medianprops)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Frame number')
ax.set_ylabel('Risk Metric')
vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax.set_xlim(0,len(R))
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
ani = animation.FuncAnimation(fig, animate, fargs=(vl,refreshPeriod), blit= True, interval=200, repeat_delay=1000, repeat=True)
plt.show()
path = results_path + 'RiskBoxPlot_' +  str(N) + '.gif' 
ani.save(path, writer='imagemagick', fps=3)


########################################################### Generate heatmap ################################################
hm_reza, hm_ppl_reza = dynamic_heatmap_model2(C_reza, framewise_positions, background)

fig = plt.figure()
display_overlay = []
for k in range(len(hm_reza)):
    display_overlay.append([plt.imshow(hm_reza[k],animated=True)])
ani = animation.ArtistAnimation(fig, display_overlay, interval=200, blit=True, repeat_delay=1000) # interval=300
plt.show()

path = results_path + 'heatmap_' +  str(N) + '.gif' 
ani.save(path, writer='imagemagick', fps=3)