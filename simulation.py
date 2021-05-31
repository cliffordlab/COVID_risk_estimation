# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:20:54 2021

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
results_path = input('Type path of folder where results should be stored: ')

#### Add or remove boundaries ####
plt.figure()
plt.title('Current plan. Want to remove any boundaries?')
plt.imshow(ep6_plan)

# NEEDS FIXING
answer = input('Do you want to remove any walls? Answer yes or no. If yes, select rectangle around the wall: ')
if answer == 'yes':
    run = True
    while run:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            run = False
            break
      
#        if keyboard.is_pressed('q'):
#            break
      
        rect = cv2.selectROI(ep6_plan_copy)
        xmin = rect[1]
        xmax = rect[1] + rect[3]
        ymin = rect[0]
        ymax = rect[0] + rect[2]
        cv2.destroyAllWindows()
        
        for i in range(xmin,xmax):
            for j in range(ymin,ymax):
                ep6_plan_copy[i,j] = [255,255,255]
                
        

if answer == 'yes':        
    plt.figure()
    plt.title('New plan with boundaries removed')
    plt.imshow(ep6_plan_copy)
    ep6_plan = ep6_plan_copy
    
# Get acceptable paths
answer = input('Do you want to make some paths unacceptable? Answer "yes" or "no": ')
if answer == 'yes':
    removed_paths = []
    for i in range(5):
        rect = cv2.selectROI(ep6_plan)
        removed_paths.append(rect)
        cv2.destroyAllWindows()
    
    # Get all pixels in accepted paths
    removed_pixels = points_inside_rect(removed_paths)
else:
    removed_pixels = None

if answer =='yes':
    ep6_plan_display = deepcopy(ep6_plan)
    for i in range(len(removed_pixels)):
        ep6_plan_display[removed_pixels[i][0], removed_pixels[i][1], :] = [255,0,0]
    plt.figure()
    plt.title('New plan with paths blocked')
    plt.imshow(ep6_plan_display)


#### Get start and end points for all people ####
N = int(input('Enter total number of people in simulation: ')) # Number of people in simulation

# Press escape after selecting region. This will close the window after recording coordinate values
# Accept starting region from user
num_start_pos = int(input('Enter number of start regions: ')) # Number of starting positions
start_areas = []
num_ppl_start = np.zeros(num_start_pos) # Number of people at each start position. Must add up to N
all_ppl_placed = False # To ensure number of people placed is total number of people
while all_ppl_placed == False:
    for i in range(num_start_pos):
        num_ppl = input('Enter number of people in this start location: ')
        num_ppl_start[i] = int(num_ppl)
        print('Number of people remaining = ', N - sum(num_ppl_start))
        rect = cv2.selectROI(ep6_plan)
        xmin_start = rect[1]
        xmax_start = rect[1] + rect[3]
        ymin_start = rect[0]
        ymax_start = rect[0] + rect[2]
        start_bounds = (xmin_start, xmax_start, ymin_start, ymax_start)
        cv2.destroyAllWindows()
        start_areas.append(start_bounds)
        if i == num_start_pos-1 and sum(num_ppl_start) == N:
            all_ppl_placed = True
        elif i == num_start_pos-1 and sum(num_ppl_start) < N:
            num_ppl_start = np.zeros(num_start_pos)
            start_areas = []
            print('ERROR: Number of people placed is less than total number of people. Start over')
        elif i == num_start_pos-1 and sum(num_ppl_start) > N:
            num_ppl_start = np.zeros(num_start_pos)
            start_areas = []
            print('ERROR: Number of people placed is more than total number of people. Start over')
        

# Accept ending region from user
num_end_pos = int(input('Enter number of end regions: '))
end_areas = []
num_ppl_end = np.zeros(num_end_pos)
all_ppl_placed = False
while all_ppl_placed == False:
    for i in range(num_end_pos):
        num_ppl = input('Enter number of people in this end location: ')
        num_ppl_end[i] = int(num_ppl)
        print('Number of people remaining = ', N - sum(num_ppl_end))
        rect = cv2.selectROI(ep6_plan)
        xmin_goal = rect[1]
        xmax_goal = rect[1] + rect[3]
        ymin_goal = rect[0]
        ymax_goal = rect[0] + rect[2]
        end_bounds = (xmin_goal, xmax_goal, ymin_goal, ymax_goal)
        cv2.destroyAllWindows()
        end_areas.append(end_bounds)
        if i == num_end_pos-1 and sum(num_ppl_end) == N:
            all_ppl_placed = True
        elif i == num_end_pos-1 and sum(num_ppl_end) < N:
            num_ppl_end = np.zeros(num_end_pos)
            end_areas = []
            print('ERROR: Number of people placed is less than total number of people. Start over')
        elif i == num_end_pos-1 and sum(num_ppl_end) > N:
            num_ppl_end = np.zeros(num_end_pos)
            end_areas = []
            print('ERROR: Number of people placed is more than total number of people. Start over')
            

#### Get user inputs for risk metric ####
random_choice = input('Do you want the algorithm to choose random values for individual\'s descriptors? Type "yes" or "no": ')
if random_choice == 'yes':
    kappa_range = input('Low, medium or high fraction of infectious particles inhaled (choose l, m, h): ')
    v_number = int(input('Number of immune people out of total: '))
    rho_range = int(input('Average age of population: '))
    zeta_range = input('Low, medium or high susceptibility due to underlying conditions (choose l, m, h): ')
    beta_range = input('Low, medium or high effectiveness of masks (choose l, m, h): ')
    eta_range = input('Low, medium or high infectiousness (choose l, m, h): ')
    
    # Low fraction of infection inhaled = between 0.1 and 0.3
    # Medium fraction of infection inhaled = between 0.4 and 0.7
    # High fraction of infection inhaled = between 0.8 and 1
    if kappa_range == 'l':
        kappa = np.array([random.randint(1,3) for x in range(N)])/10
    elif kappa_range == 'm':
        kappa = np.array([random.randint(4,7) for x in range(N)])/10
    else:
        kappa = np.array([random.randint(8,10) for x in range(N)])/10
     
    # Randomly selecting v_number of immune people
    v = np.ones(N)
    immune_indices = random.sample(range(0,N), v_number)
    v[immune_indices] = 0
    
    # Age groups:              0-4, 5-17, 18-29, 30-39, 40-49, 50-64, 65-74, 75-84, 85+
    # Risk of hospitalization: 2x,  1x,   6x,    10x,   15x,   25x,   40x,   65x,   95x
    # Ref: https://www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-age.html
    if rho_range >= 0 and rho_range <= 4:
        rho = 2*np.ones(N)/259
    elif rho_range > 4 and rho_range <= 17:
        rho = 1*np.ones(N)/259
    elif rho_range > 17 and rho_range <= 29:
        rho = 6*np.ones(N)/259
    elif rho_range > 29 and rho_range <= 39:
        rho = 10*np.ones(N)/259
    elif rho_range > 39 and rho_range <= 49:
        rho = 15*np.ones(N)/259
    elif rho_range > 49 and rho_range <= 64:
        rho = 25*np.ones(N)/259
    elif rho_range > 64 and rho_range <= 74:
        rho = 40*np.ones(N)/259
    elif rho_range > 74 and rho_range <= 84:
        rho = 65*np.ones(N)/259
    elif rho_range > 84:
        rho = 95*np.ones(N)/259
        
    # Low susceptibility due to underlying condition = between 0.1 and 0.3
    # Medium susceptibility due to underlying condition = between 0.4 and 0.7
    # High susceptibility due to underlying condition = between 0.8 and 1
    if zeta_range == 'l':
        zeta = np.array([random.randint(1,3) for x in range(N)])/10
    elif zeta_range == 'm':
        zeta = np.array([random.randint(4,7) for x in range(N)])/10
    else:
        zeta = np.array([random.randint(8,10) for x in range(N)])/10
        
    # Low effectiveness of mask = between 0.1 and 0.3
    # Medium effectiveness of mask = between 0.4 and 0.7
    # High effectiveness of mask = between 0.8 and 1
    if beta_range == 'l':
        beta = np.array([random.randint(1,3) for x in range(N)])/10
    elif beta_range == 'm':
        beta = np.array([random.randint(4,7) for x in range(N)])/10
    else:
        beta = np.array([random.randint(8,10) for x in range(N)])/10
        
    # Low infectiousness = between 0.1 and 0.3
    # Medium infectiousness = between 0.4 and 0.7
    # Highinfectiousness = between 0.8 and 1
    if eta_range == 'l':
        eta = np.array([random.randint(1,3) for x in range(N)])/10
    elif eta_range == 'm':
        eta = np.array([random.randint(4,7) for x in range(N)])/10
    else:
        eta = np.array([random.randint(8,10) for x in range(N)])/10
else:
    kappa = np.array(eval('[' + input('Fraction of infectious particles inhaled per person: ') + ']')[0])*np.ones(N)
    v = np.array(eval('[' + input('Immune (0) or not immune (1) per person: ') + ']')[0])*np.ones(N)
    rho_range = np.array(eval('[' + input('Age of each person: ') + ']')[0])*np.ones(N)
    zeta = np.array(eval('[' + input('Susceptibility due to underlying conditions per person: ') + ']')[0])*np.ones(N)
    beta = np.array(eval('[' + input('Effectiveness of masks per person: ') + ']')[0])*np.ones(N)
    eta = np.array(eval('[' + input('Infectiousness per person: ') + ']')[0])*np.ones(N)
    
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
    
air_change_percent = float(input('Percentage of air changed every T minutes: '))/100
air_change_rate = int((input('T in minutes from last question: ')))*60
hl = float(input('Half life, i.e. 50% of particles decay in how many minutes: '))*60

#### Get MCIness of people ####
random_choice = input('Do you want the algorithm to choose random values for individual\'s MCIness? Type "yes" or "no": ')
if random_choice == 'yes':
    mu_range = input('Low, medium or high MCIness (choose l, m, h): ')
    
    # Healthy: mu = between 0.1 and 0.3 
    # Mild cognitive impairment: mu = between 0.4 and 0.7
    # Moderate cognitive impairment: mu = between 0.8 and 1
    if mu_range == 'l':
        mu = np.array([random.randint(1,3) for x in range(N)])/10
    elif mu_range == 'm':
        mu = np.array([random.randint(4,7) for x in range(N)])/10
    else:
        mu = np.array([random.randint(8,10) for x in range(N)])/10
else:
    mu = (eval('[' + input('MCIness per person: ') + ']')[0])*np.ones(N)
    
    
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get paths of N people using RRT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

# Get random, specific positions from areas specified above
startPos = []
endPos = []
for p in range(num_start_pos):
    for n in range(int(num_ppl_start[p])):
        s = random_node(start_areas[p][0], start_areas[p][1], start_areas[p][2], start_areas[p][3], obstacles)
        startPos.append(s)
for p in range(num_end_pos):
    for n in range(int(num_ppl_end[p])):
        e = random_node(end_areas[p][0], end_areas[p][1], end_areas[p][2], end_areas[p][3], obstacles)
        endPos.append(e)

# Each frame is around 2 seconds
tolerance = 5
maxStep = 20
allPaths = []
Ts = [] # Sampling rate for frame
for n in range(N):
    print('Calculating path of person ', n)
    path, check_int = RRT_full_path(endPos[n], startPos[n], obstacles, mu[n], tolerance, ep6_plan, removed_pixels)
    #path = add_pause(path, 20, 0)
    allPaths.append(path)

# Save paths as numpy file
path_path = results_path + 'path.npy'
np.save(path_path, allPaths)

# Load paths if necessary
allPaths = np.load(results_path+'path.npy', allow_pickle=True)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Make all paths of same length
maxLen = 0
for n in range(N):
    if len(allPaths[n]) > maxLen:
        maxLen = len(allPaths[n])

for n in range(N):
    if len(allPaths[n]) < maxLen:
        diff = maxLen - len(allPaths[n])
        for i in range(diff):
            allPaths[n] = np.concatenate((allPaths[n], [[allPaths[n][-1][0], allPaths[n][-1][1]]]), 0)
            
frames = [] 
ep6_plan_copy1 = copy.deepcopy(ep6_plan) 
if removed_pixels != None:
    for i in range(np.shape(ep6_plan_copy)[0]):
        for j in range(np.shape(ep6_plan_copy)[1]):
            if (ep6_plan_copy1[i,j] == [0,0,0]).all() or (i,j) in removed_pixels:
                pass
            else:
                ep6_plan_copy1[i,j] = [200,0,0]
                
for t in range(maxLen):
    ep6_plan_copy = copy.deepcopy(ep6_plan_copy1)
    
    for p in range(num_start_pos): # Display starting region
        ep6_plan_copy[start_areas[p][0]:start_areas[p][1], start_areas[p][2]:start_areas[p][3], 0] = 0
    for p in range(num_end_pos): # Display ending region
        ep6_plan_copy[end_areas[p][0]:end_areas[p][1], end_areas[p][2]:end_areas[p][3], 2] = 0
    
    for n in range(N):
        ep6_plan_copy[allPaths[n][t][0]-2:allPaths[n][t][0]+2, allPaths[n][t][1]-2:allPaths[n][t][1]+2, 1] = 0
    frames.append(ep6_plan_copy)
    
fig = plt.figure()
display_overlay = []
for k in range(maxLen):
    display_overlay.append([plt.imshow(frames[k],animated=True)])
ani = animation.ArtistAnimation(fig, display_overlay, interval=200, blit=True, repeat_delay=1000) # interval=300
plt.show()  
    

# Save animation
path = results_path + 'pplWalking_' + str(N) +  '.gif' 
ani.save(path, writer='imagemagick', fps=3)
# Save as video
Writer = animation.writers['ffmpeg']
writer = Writer(fps=6, metadata=dict(artist='Me'), bitrate=1800)
path = 'results/ep6_day/break_to_kitchen/' + str(N) + '_' + 'pplWalking.mp4'
ani.save(path, writer=writer)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get social distancing metric values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
alpha = 8 # alpha = 2m. 1 pixel = 0.26m, therefore approx. 8 pixels = 2m.
sd_metric_values, framewise_positions, all_distances = get_metric_values(allPaths, maxLen, alpha)

# Get SD metric for each person
sd_personwise = []
for i in range(len(framewise_positions)):
    sd_temp = []
    for n in range(N):
        sd = sd_metric_per_person(framewise_positions[i], n, alpha)
        sd_temp.append(sd)
    sd_personwise.append(sd_temp)
    

##### Box plot of distances #####
duration = 20 # in sec
refreshPeriod = 1000 # in ms
x = np.arange(1,len(all_distances)+1,5)
xlabel = []
for i in range(len(all_distances)):
    if i%5 == 0:
        xlabel.append(i+1)
fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(all_distances, showfliers=False, medianprops=medianprops)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Frame number')
ax.set_ylabel('Distances in m')
vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax.set_xlim(0,len(all_distances))
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
ani = animation.FuncAnimation(fig, animate, fargs=(vl,refreshPeriod), blit= True, interval=200, repeat_delay=1000, repeat=True)
plt.show()
path = results_path + 'distBoxPlot_' + str(N) +  '.gif' 
ani.save(path, writer='imagemagick', fps=3)
# Save as video
Writer = animation.writers['ffmpeg']
writer = Writer(fps=6, metadata=dict(artist='Me'), bitrate=1800)
path = 'results/ep6_day/break_to_kitchen/' + str(N) + '_' + 'distBoxPlot.mp4'
ani.save(path, writer=writer)


# Box plot of SD metric for each individual
duration = 20 # in sec
refreshPeriod = 1000 # in ms
x = np.arange(1,len(sd_personwise)+1,5)
xlabel = []
for i in range(len(sd_personwise)):
    if i%5 == 0:
        xlabel.append(i+1)
fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(sd_personwise, showfliers=False, medianprops=medianprops)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Frame number')
ax.set_ylabel('SD metric value')
vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax.set_xlim(0,len(all_distances))
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
ani = animation.FuncAnimation(fig, animate, fargs=(vl,refreshPeriod), blit= True, interval=200, repeat_delay=1000, repeat=True)
plt.show()
path = results_path + 'SDmetricBoxPlot_' + str(N) + '.gif' 
ani.save(path, writer='imagemagick', fps=3)
# Save as video
Writer = animation.writers['ffmpeg']
writer = Writer(fps=6, metadata=dict(artist='Me'), bitrate=1800)
path = 'results/ep6_day/break_to_kitchen/' + str(N) + '_' + 'sdMetricIndividual.mp4'
ani.save(path, writer=writer)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get risk metric %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Extend frames to meet F_s requirements
#F_s = 0.006 #number of secconds in one frame
#interstitial_frame_num =  round(1/F_s)
#allPaths_extended = []
#for i in range(len(allPaths)):
#    for j in range(len(allPaths[i])):
##        x = np.linspace(allPaths[i][j][0], allPaths[i][j+1][0], interstitial_frame_num)
##        y = np.linspace(allPaths[i][j][1], allPaths[i][j+1][1], interstitial_frame_num)
#        x = (allPaths[i][j][0]*np.ones(interstitial_frame_num)).astype(int)
#        y = (allPaths[i][j][1]*np.ones(interstitial_frame_num)).astype(int)
#        if j==0:
#            X = x
#            Y = y
#        else:
#            X = np.concatenate((X,x))
#            Y = np.concatenate((Y,y))
#    paths_temp = np.vstack((X,Y)).T
#    allPaths_extended.append(paths_temp)
        
#maxLen = len(allPaths_extended[0])
#_, framewise_positions_extended, _ = get_metric_values(allPaths_extended, maxLen, alpha)

R_reza, C_reza, pixels_to_color, final_inf = risk_detailed_reza_original(framewise_positions, kappa=kappa, v=v, rho=rho, 
                                    zeta=zeta, beta=beta, eta=eta, mu=1, D=0.003, air_change_percent=air_change_percent, 
                                    air_change_rate=air_change_rate, neighbor_dist=1,  F_s=1, hl=hl, num_initial_inf=6,
                                    ep6_plan=ep6_plan, XMIN1=12, XMAX1=115, YMIN1=24, YMAX1=210)

r_path = results_path + 'R.npy'
np.save(r_path, R_reza)
c_path = results_path + 'C.npy'
np.save(c_path, C_reza)
#
## Load R and C if necessary
#R_reza = np.load(results_path+'R.npy')
#C_reza = np.load(results_path+'C.npy')

# Box plot for risk metric
R = []
R_avg = []
for i in range(len(R_reza)):
    R.append(R_reza[i])
    R_avg.append(np.mean(R_reza[i]))
duration = 20 # in sec
refreshPeriod = 1000 # in ms
x = np.arange(1,len(R_reza)+1, 5)
xlabel = []
for i in range(len(sd_personwise)):
    if i%5 == 0:
        xlabel.append(i+1)
fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(R, showfliers=False, medianprops=medianprops)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Frame number')
ax.set_ylabel('Risk Metric')
vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax.set_xlim(0,len(all_distances))
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
ani = animation.FuncAnimation(fig, animate, fargs=(vl,refreshPeriod), blit= True, interval=200, repeat_delay=1000, repeat=True)
plt.show()
path = results_path + 'RiskBoxPlotAnimated_' +  str(N) + '.gif' 
ani.save(path, writer='imagemagick', fps=3)

fig,ax = plt.subplots()
medianprops = dict(linestyle='-.', linewidth=4.5, color='firebrick')
ax.boxplot(R, showfliers=False, medianprops=medianprops)
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Frame number')
ax.set_ylabel('Risk Metric')
ax.set_xlim(0,len(all_distances))
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
plt.show()
fig.set_size_inches((15, 7), forward=False)
path = results_path + 'RiskBoxPlot_ep6_vaccine70.png'
plt.savefig(path)

# Save as video
Writer = animation.writers['ffmpeg']
writer = Writer(fps=6, metadata=dict(artist='Me'), bitrate=1800)
path = 'results/ep6_day/break_to_kitchen/' + str(N) + '_' + 'riskBoxPlot.mp4'
ani.save(path, writer=writer)


# Plot combine avg risk value
plt.figure()
plt.plot(R_avg20, label='20% air change')
plt.plot(R_avg50, label='50% air change')
plt.plot(R_avg80, label='80% air change')
plt.xticks(x, xlabel, size=20)
plt.yticks(size=20)
ax.set_xlabel('Frame number')
ax.set_ylabel('Risk Metric')
plt.legend()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get heatmap %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
hm_reza, hm_ppl_reza = dynamic_heatmap_model2(C_reza, framewise_positions, ep6_plan)

fig = plt.figure()
display_overlay = []
for k in range(len(hm_reza)):
    #if k % interstitial_frame_num == 0:
    display_overlay.append([plt.imshow(hm_ppl_reza[k],animated=True)])
ani = animation.ArtistAnimation(fig, display_overlay, interval=1000, blit=True, repeat_delay=1000) # interval=300
plt.show()

path = results_path + 'heatmap_' +  str(N) + '.gif' 
ani.save(path, writer='imagemagick', fps=3)
# Save as video
Writer = animation.writers['ffmpeg']
writer = Writer(fps=6, metadata=dict(artist='Me'), bitrate=1800)
path = 'results/ep6_day/break_to_kitchen/' + str(N) + '_' + 'heatmap.mp4'
ani.save(path, writer=writer)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Determine superspreading event %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Place people randomly in building
answer = input('Do you want to make some paths unacceptable? Answer "yes" or "no": ')
if answer == 'yes':
    removed_paths = []
    for i in range(5):
        rect = cv2.selectROI(ep6_plan)
        removed_paths.append(rect)
        cv2.destroyAllWindows()
    
    # Get all pixels in accepted paths
    removed_pixels_sse = points_inside_rect(removed_paths)
else:
    removed_pixels_sse = None
    
    
def random_point(removed_pixels=None):
    
    # Get all points points d steps away
    points = []
    
    # Select one point from all points uniformly distributed
    ind = random.randint(0,len(points)-1)
    (x_rand, y_rand) = points[ind]
    
    # Check if point lies in building bounds and in accepted_pixels
    if removed_pixels == None:
        if x_rand >= XMIN1 and x_rand <= XMAX1 and y_rand >= YMIN1 and y_rand <= YMAX1:
            return ((x_rand, y_rand))
        else:
            return(random_point(removed_pixels))
    else:
        if x_rand >= XMIN1 and x_rand <= XMAX1 and y_rand >= YMIN1 and y_rand <= YMAX1 and (x_rand, y_rand) not in removed_pixels:
            return ((x_rand, y_rand))
        else:
            return(random_point(removed_pixels))

  
    
XMIN1 = 12 #15
XMAX1 = 115 #111
YMIN1 = 24 #22
YMAX1 = 210 #198
static_positions = []
for n in range(N):
    x = np.random.randint(XMIN1,XMAX1)
    y = np.random.randint(YMIN1,YMAX1)
    while (x,y) in obstacles or (x,y) in removed_pixels_sse:
        x = np.random.randint(XMIN1,XMAX1)
        y = np.random.randint(YMIN1,YMAX1)
    static_positions.append((x,y))

# Visualize positions
ep6_sse = copy.deepcopy(ep6_plan)
for n in range(N):
    x,y = static_positions[n]
    ep6_sse[x-2:x+2,y-2:y+2,:] = [255,0,0]
plt.imshow(ep6_sse)

# Extend positions for more frames
#sim_time = 1500 # simulation time in seconds
#num_frames = int(sim_time/F_s)
#static_framewise_positions = []
#for i in range(num_frames):
static_framewise_positions = static_positions
    
# Decide total time of simulation in seconds
init_num_infected = int(0.1*N) # 20% people infected at first
#R_reza, C_reza, pixels_to_color, infection_final = risk_detailed_reza_original(static_framewise_positions, kappa=kappa, v=v, rho=rho, 
#                                    zeta=zeta, beta=beta, eta=eta, mu=1, D=50*10**-2, air_change_percent=air_change_percent, 
#                                    air_change_rate=air_change_rate, neighbor_dist=1,  F_s=F_s, hl=hl, num_initial_inf=init_num_infected,
#                                    ep6_plan=ep6_plan, XMIN1=12, XMAX1=115, YMIN1=24, YMAX1=210)

sse_time, inf_num = risk_sse(pos=static_framewise_positions, kappa=kappa, v=v, rho=rho, 
                                    zeta=zeta, beta=beta, eta=eta, mu=1, D=0.003, air_change_percent=air_change_percent, 
                                    air_change_rate=air_change_rate, neighbor_dist=1,  F_s=1, hl=hl, num_initial_inf=init_num_infected,
                                    ep6_plan=ep6_plan, XMIN1=12, XMAX1=115, YMIN1=24, YMAX1=210)

print(sse_time)

#hm_reza, hm_ppl_reza = dynamic_heatmap_model2(C_reza, static_framewise_positions, ep6_plan)
#
#fig = plt.figure()
#display_overlay = []
#for k in range(len(hm_reza)):
#    if k%400==0:
#        display_overlay.append([plt.imshow(hm_reza[k],animated=True)])
#ani = animation.ArtistAnimation(fig, display_overlay, interval=200, blit=True, repeat_delay=1000) # interval=300
#plt.show()

num_infected = 0
for i in range(len(inf_num)):
    if inf_num[i] >= 10**7/10**9:
        num_infected += 1
print(num_infected)