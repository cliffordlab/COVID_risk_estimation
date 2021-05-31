# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 01:30:33 2021

@author: chait
"""

import numpy as np
from utils import check_wall_between_ppl
import scipy.signal
import matplotlib.pyplot as plt

def distancing_metric(distances, alpha, num_people_frames):
    St_frames = []    
    St = 0        
    if distances != []:
        for i in range(len(distances)):
            St += np.exp(-(distances[i]**2-alpha**2)/alpha**2)
        St_frames.append(St)
    else:
            St_frames.append(-3)
            
    return St_frames


def get_metric_values(allPaths, T, alpha): 
    pos = []
    N = len(allPaths)
    for t in range(T):
        temp = []
        for n in range(N):
            try:
                temp.append(allPaths[n][t])
            except:
                temp.append(allPaths[n][-1])
        pos.append(temp)
        
    avg_sd = []
    avg_dist = []
    all_dist = [] # Stores distances between each pair of individuals for each frames
    for t in range(len(pos)):
        dist = []
        dist_meters = []
        for i in range(len(pos[t])):
            for j in range(i+1,len(pos[t])):
                d = np.sqrt((pos[t][i][0]-pos[t][j][0])**2 + (pos[t][j][1]-pos[t][j][1])**2) # Get distance b/w each person
                dist.append(d)
                dist_meters.append(0.26*d) # Multiplied by 0.26 to convert from pixels to meters
        all_dist.append(dist_meters) 
        avg_dist.append(np.mean(dist))
        sd = distancing_metric(dist, alpha, N)
        avg_sd.append(sd)
    avg_sd = np.array(avg_sd)
    return avg_sd, pos, all_dist

def sd_metric_per_person(pos, person_ind, alpha):
    # Checks social distancing metric between one person and all ppl around him
    
    sd_metric = 0
    pos1 = pos[person_ind]
    for p in range(len(pos)):
        if p != person_ind:
            pos2 = pos[p]
            dist = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
            sd_metric += np.exp(-(dist**2-alpha**2)/alpha**2)
    
    return sd_metric

######################################################### Detailed risk model ######################################################
XMIN1 = 12 #15
XMAX1 = 115 #111
YMIN1 = 24 #22
YMAX1 = 209 #198

def pts_within_circles(rad1, rad2, ep6_plan, spatial_diff_rate, x0, y0):
    # rad2 > rad1
    if rad2<=rad1:
        print('Error in circle radii. radius 2 must be greater than radius 2')
    else:
        pts = []
        X=[]
        Y=[]
        num_pts = 0
        coords_weights = []
        sum_of_weights = 0
        for x in range(x0-rad2,x0+rad2):
            for y in range(y0-rad2, y0+rad2):
                if x >= XMIN1 and x < XMAX1 and y >= YMIN1 and y < YMAX1:
                    pixel_intersect = check_wall_between_ppl((x0,y0), (x,y), ep6_plan)
                    if (x-x0)**2 + (y-y0)**2 >= rad1 and (x-x0)**2 + (y-y0)**2 <= rad2 and pixel_intersect==False:
                        X.append(x)
                        Y.append(y)
                        pts.append((y,x))
                        num_pts += 1
                        coords_weights.append((y, x, np.exp(-((x-x0)**2 + (y-y0)**2)/spatial_diff_rate)))
                        sum_of_weights += np.exp(-((x-x0)**2 + (y-y0)**2)/spatial_diff_rate)
        #pts = [X,Y]
    return np.array(pts), num_pts, coords_weights, sum_of_weights

def saturate(Ci, mu):
    Ci = np.tanh(mu*Ci)
    return Ci

XMIN1 = 12 #15
XMAX1 = 115 #111
YMIN1 = 24 #22
YMAX1 = 210 #198
#def diffusion(Ci, Ci_prev, ppl_pos, D, lamb, F_s, neighbor_centers, neighbor_dist, ep6_plan, XMIN1=12, XMAX1=115, YMIN1=24, YMAX1=210):
#    num_neigh=neighbor_dist
#    Ci_diff = np.zeros_like(Ci)
#    
#    for i in range(np.shape(Ci)[0]):
#        for j in range(np.shape(Ci)[1]):
#                
#            continue_code = False
#            if Ci[i,j] != 0:
#                for p in ppl_pos:
#                    print(p, (i,j))
#                    if ((i,j) == np.array(p)).all():
#                        continue_code = True
#                        print(continue_code)
#                        break
#                        
#            if continue_code == True:
#        
#                if (ep6_plan[i,j] > [210,210,210]).all():
#                
#                    neighbors = []
#                    for n1 in range(-num_neigh,num_neigh+1):
#                        for n2 in range(-num_neigh,num_neigh+1):
#                            if i+n1 >= XMIN1 and i+n1 < XMAX1 and j+n2 >= YMIN1 and j+n2 < YMAX1:
#                                if check_wall_between_ppl((j,i), (j+n2,i+n1), ep6_plan)==False \
#                                    and (ep6_plan[i+n1,j+n2]>[210,210,210]).all():
#                                    neighbors.append((i+n1,j+n2))
#                    
#                    if neighbors != []:
#                        w = np.zeros(len(neighbors)-1)
#                        for k in range(len(neighbors)-1):
#                            if neighbors[k][0] >= 0 and neighbors[k][0] < 138 and neighbors[k][1] >= 0 and neighbors[k][1] < 226 and \
#                            (neighbors[k] != (i,j)):
#                                w[k] = D*F_s/(np.sqrt((i-neighbors[k][0])**2 + (j-neighbors[k][1])**2))
#                            
#                        w_mid = 1 - sum(w)
#                        if w_mid < 0:
#                            print('Error: Choose smaller D')
#                            break
#                        
#                        W = sum(w)+w_mid
#                        
#                        summ = 0
#                        for k in range(len(neighbors)-1):
#                            if neighbors[k][0] >= 0 and neighbors[k][0] < 138 and neighbors[k][1] >= 0 and neighbors[k][1] < 226 and \
#                            (neighbors[k] != (i,j)):
#                                summ += w[k]*Ci[int(neighbors[k][0]), int(neighbors[k][1])]/W
#                        summ += w_mid*Ci[i,j]/W
#                        Ci_diff[i,j] = summ
#        
#    return (lamb)*Ci_diff

room_path = 'ep6_outline_smallest.jpg'
ep6_plan = plt.imread(room_path)
neighbor_plan = 255*np.ones_like(ep6_plan)
for i in range(np.shape(ep6_plan)[0]):
    for j in range(np.shape(ep6_plan)[1]):
        if ep6_plan[i,j,0] < 50:
            neighbor_plan[i-1,j,:] = [0,0,0]
            neighbor_plan[i-1,j-1,:] = [0,0,0]
            neighbor_plan[i-1,j+1,:] = [0,0,0]
            neighbor_plan[i,j,:] = [0,0,0]
            neighbor_plan[i,j-1,:] = [0,0,0]
            neighbor_plan[i,j+1,:] = [0,0,0]
            neighbor_plan[i+1,j,:] = [0,0,0]
            neighbor_plan[i+1,j-1,:] = [0,0,0]
            neighbor_plan[i+1,j+1,:] = [0,0,0]
            
# Try 2 - conv
def diffusion(Ci, D, lamb, F_s, neighbor_dist, ep6_plan, XMIN1=12, XMAX1=115, YMIN1=24, YMAX1=210):
    
    sliding_window = np.zeros((neighbor_dist*2+1 ,neighbor_dist*2+1))
    center = neighbor_dist
    w = []
    for i in range(np.shape(sliding_window)[0]):
        for j in range(np.shape(sliding_window)[1]):
            if (i,j) != (center,center):
                dist = ((i-center)**2 + (j-center)**2)*0.26**2
                sliding_window[i,j] = D*F_s/dist
                w.append(D*F_s/dist)
    if sum(w) >= 1:
        print('Error: Choose smaller D')
        return
    else:
        sliding_window[center,center] = 1-sum(w)
        
    # 2D convolution
    #Ci_diff = scipy.signal.convolve2d(Ci, sliding_window, mode='same', boundary='symm')
    #Ci_diff = np.where(neighbor_plan[:,:,0]>=200, scipy.signal.convolve2d(Ci, sliding_window, mode='same', boundary='symm'), 0)
    Ci_diff = scipy.signal.convolve2d(Ci, sliding_window, mode='same', boundary='symm')
        
    return (lamb)*Ci_diff

#def diffusion2(Ci, D, lamb, F_s, neighbor_centers, neighbor_dist, ep6_plan, shape='circle'):
#    num_neigh=neighbor_dist
#    Ci_diff = np.zeros_like(Ci)
#    alpha_x = D*0.01/0.1**2
#    alpha_y = D*0.01/0.1**2
#    alpha_xy = D*0.01/(0.1**2 + 0.1**2)
#    
#    # Check validity of the parameters
#    if (1 - 2 * alpha_x - 2 * alpha_y) < 0 or (1 - 2 * alpha_x - 2 * alpha_y - 2 * alpha_xy - 2 * alpha_xy) < 0:
#        print('Stability condition for parameters not fulfilled. Make simulation time period smaller')
#    
#    for i in range(np.shape(Ci)[0]):
#        for j in range(np.shape(Ci)[1]):
#            try:
#                Ci_diff[i,j] = (1 - 2*alpha_x - 2*alpha_y - 2*alpha_xy - 2*alpha_xy)*Ci[i,j] + alpha_x*Ci[i-1,j] + \
#                            alpha_x*Ci[i+1,j] + alpha_y*Ci[i,j-1] + alpha_y*Ci[i,j+1] + alpha_xy*Ci[i-1,j-1] + \
#                            alpha_xy*Ci[i+1,j+1] + alpha_xy*Ci[i-1,j+1] + alpha_xy*Ci[i+1,j-1]
#            except:
#                pass
#            
#    return (lamb)*Ci_diff


def risk_detailed_reza_original(pos, kappa, v, rho, zeta, beta, eta, mu, D, air_change_percent, air_change_rate, neighbor_dist,
                       F_s, hl, num_initial_inf, ep6_plan, XMIN1=12, XMAX1=115, YMIN1=24, YMAX1=210):
    
    # Convert neighborhood distance to pixels from meters
    neighbor_dist = int(round(neighbor_dist/0.26))
    # Individualized risk factors
    sigma = v*rho*zeta
    
    # Vanishing rate of air particles/infectious particles due to air change and decay
    lamb = 0.5**(F_s/hl) * air_change_percent**(F_s/air_change_rate)
        
    # Concentration of infection particles matrix
    C = np.zeros((len(pos), np.shape(ep6_plan)[0], np.shape(ep6_plan)[1]))
    pixels_to_color = np.zeros((np.shape(ep6_plan)[0], np.shape(ep6_plan)[1]))
    
    I = np.zeros((len(pos),len(pos[0])))
    R = np.zeros((len(pos),len(pos[0])))
    
    #I[0,:] = np.random.rand(len(pos[0]))
    I[0,:] = np.ones(len(pos[0]))/10**9
    inf_individuals = np.random.randint(0,len(pos[0]), num_initial_inf)
    #inf_individuals=num_initial_inf
    I[0,inf_individuals] = 10**9/10**9
    #I[0,:] = 10**4*np.ones(len(pos[0]))
    R[0,:] = np.tanh(sigma*I[0,:])
    E = beta*eta*R[0,:]
    R_test = np.zeros_like(R)
    for i in range(len(pos)):
    
        print('Processing frame ', i, ' of ', len(pos))
        neighbor_centers = []
        for j in range(len(pos[i])):
            y,x = pos[i][j]
            neighbor_centers.append((y,x))
            C[i,y,x] = C[i,y,x] + E[j]
        
#        C[i] = diffusion(C[i], C[i], pos[i], D, lamb, F_s, neighbor_centers, neighbor_dist, ep6_plan, XMIN1, XMAX1, 
#         YMIN1, YMAX1)

        C[i] = diffusion(C[i], D, lamb, F_s, neighbor_dist, ep6_plan, XMIN1, XMAX1, 
         YMIN1, YMAX1)
        
        if i < len(pos)-1:
            C[i+1] = saturate(C[i],mu)
        
        for j in range(len(pos[i])):
            if i < len(pos)-1:
                y,x = pos[i][j]
                I[i+1,j] = lamb*I[i,j] + kappa[j]*C[i+1,y,x]
                if I[i+1,j] < 0:
                    print(I[i+1,j])
                R[i+1,j] = np.tanh(sigma[j]*I[i+1,j])
                E[j] = beta[j]*eta[j]*R[i+1,j]
                R_test[i,j] = sigma[j]*C[i,y,x]
        
    return R, C, pixels_to_color, I[-1,:]


def risk_sse(pos, kappa, v, rho, zeta, beta, eta, mu, D, air_change_percent, air_change_rate, neighbor_dist,
                       F_s, hl, num_initial_inf, ep6_plan, XMIN1=12, XMAX1=115, YMIN1=24, YMAX1=210):
    
    # Convert neighborhood distance to pixels from meters
    neighbor_dist = int(round(neighbor_dist/0.26))
    # Individualized risk factors
    sigma = v*rho*zeta
    
    # Vanishing rate of air particles/infectious particles due to air change and decay
    lamb = 0.5**(F_s/hl) * air_change_percent**(F_s/air_change_rate)
        
    # Concentration of infection particles matrix
    C_curr = np.zeros((np.shape(ep6_plan)[0], np.shape(ep6_plan)[1]))
    C_next = np.zeros((np.shape(ep6_plan)[0], np.shape(ep6_plan)[1]))
    
    I_curr = np.zeros(len(pos))
    I_next = np.zeros(len(pos))
    R_curr = np.zeros(len(pos))
    R_next = np.zeros(len(pos))
    
    #I[0,:] = np.random.rand(len(pos[0]))
    I_curr = np.ones(len(pos))/10**9
    inf_individuals = np.random.randint(0,len(pos), num_initial_inf)
    I_curr[inf_individuals] = 10**94/10**9
    #I[0,:] = 10**4*np.ones(len(pos[0]))
    R_curr = np.tanh(sigma*I_curr)
    E = beta*eta*R_curr
    sse_time = 0
    i=0
    while sse_time==0:
    
        print('Processing frame ', i)
        neighbor_centers = []
        for j in range(len(pos)):
            y,x = pos[j]
            neighbor_centers.append((y,x))
            C_curr[y,x] = C_curr[y,x] + E[j]

        C_curr = diffusion(C_curr, D, lamb, F_s, neighbor_dist, ep6_plan, XMIN1, XMAX1, 
         YMIN1, YMAX1)
        
        #if i < len(pos)-1:
        C_next = saturate(C_curr,mu)
        
        infectious_num = 0
        for j in range(len(pos)):
            #if i < len(pos)-1:
            y,x = pos[j]
            I_next[j] = lamb*I_curr[j] + kappa[j]*C_next[y,x]
            if I_next[j] < 0:
                print(I_next[j])
            R_next[j] = np.tanh(sigma[j]*I_next[j])
            E[j] = beta[j]*eta[j]*R_next[j]
            if I_next[j] >= 10**9/10**9:
                infectious_num += 1
        
        if infectious_num >= int(len(pos)*0.8) and sse_time == 0:
            sse_time = i
        i += 1
        
        R_curr=R_next
        I_curr=I_next
        C_curr=C_next
        
    return sse_time, I_next