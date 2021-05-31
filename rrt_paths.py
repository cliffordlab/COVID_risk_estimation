# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 00:57:10 2021

@author: chait
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:06:28 2020

@author: chait
"""

import numpy as np
from utils import *
from probability_distribution import *
import random
#from anytree import Node, RenderTree, resolver
#from anytree.exporter import DotExporter
import time
#from random import randint as ri
#from multiprocessing import Pool


# Limits of image or building boundary of image - TO DO: Select from user by automatically clicking corners
XMIN = 27
XMAX = 184
YMIN = 37
YMAX = 330

XMIN1 = 12 #15
XMAX1 = 115 #111
YMIN1 = 24 #22
YMAX1 = 209 #198


def random_point(prev_pos, mci_weights, mci_mu, mci_sig, removed_pixels=None):
    # Get path length
    path_len = path_length(mci_weights, mci_mu, mci_sig)
    
    if path_len == 0:
        return ((prev_pos[0],prev_pos[1]))
    
    # Get all points points d steps away
    points = []
    xmin = max(XMIN, prev_pos[0]-path_len)
    xmax = min(XMAX, prev_pos[0]+path_len)
    ymin = max(YMIN, prev_pos[1]-path_len)
    ymax = min(YMAX, prev_pos[1]+path_len)
    for i in range(xmin,xmax):
        for j in range(ymin,ymax):
            pt = (i,j)
            points.append(pt)
    if points == []: # If no points found, then run again with different path length
        return ((prev_pos[0],prev_pos[1]))
        
    # Select one point from all points uniformly distributed
    ind = random.randint(0,len(points)-1)
    (x_rand, y_rand) = points[ind]
    
    # Check if point lies in building bounds and in accepted_pixels
    if removed_pixels == None:
        if x_rand >= XMIN1 and x_rand <= XMAX1 and y_rand >= YMIN1 and y_rand <= YMAX1:
            return ((x_rand, y_rand))
        else:
            return(random_point(prev_pos, mci_weights, mci_mu, mci_sig, removed_pixels))
    else:
        if x_rand >= XMIN1 and x_rand <= XMAX1 and y_rand >= YMIN1 and y_rand <= YMAX1 and (x_rand, y_rand) not in removed_pixels:
            return ((x_rand, y_rand))
        else:
            return(random_point(prev_pos, mci_weights, mci_mu, mci_sig, removed_pixels))


def RRT(x,y,parent,obstacles,Step, ep6_plan):
    if (x,y) not in parent and (x,y) not in obstacles:
        x_m,y_m=-1,-1
        cur_min=100000000000000
        got_a_v = False
        for v in parent:
            intersect = pass_thru_obstacle(x,y,v[0],v[1],ep6_plan)
            if p2p_dist(v,(x,y))<cur_min and pass_thru_obstacle(x,y,v[0],v[1],ep6_plan) == False:
                x_m,y_m=v
                cur_min =  p2p_dist(v,(x,y))
                got_a_v = True

        if got_a_v == True:
            good = True
            ans=[]
            if abs(x_m - x)<abs(y_m-y):
                if y_m<y:
                    for u in range(y_m+1, y+1):
                        x_cur = int (((x_m - x)/(y_m - y))*( u - y) + x)
                        y_cur = u
                        if (x_cur,y_cur) in obstacles:
                            good=False
                            break
                    if good:
                        ans=[int (((x_m - x)/(y_m - y))*( y_m+Step - y) + x),y_m+Step]
                else:
                    for u in range(y, y_m):
                        x_cur = int(((x_m - x)/(y_m - y))*( u - y) + x)
                        y_cur = u
                        if (x_cur,y_cur) in obstacles:
                            good=False
                            break
                    if good:
                        ans=[int (((x_m - x)/(y_m - y))*( y_m-Step - y) + x),y_m-Step]
    
            else:
                if x_m<x:
                    for u in range(x_m + 1, x+1):
                        x_cur = u
                        y_cur = int( ((y_m-y)/(x_m-x))*(u-x) + y )
                        if (x_cur,y_cur) in obstacles:
                            good=False
                            break
                    if good:
                        ans=[x_m+Step,int( ((y_m-y)/(x_m-x))*(x_m+Step-x) + y ) ]
                else:
                    for u in range(x , x_m):
                        x_cur = u
                        y_cur = int( ((y_m-y)/(x_m-x))*(u-x) + y )
                        if (x_cur,y_cur) in obstacles:
                            good=False
                            break
                    if good:
                        ans=[x_m-Step,int( ((y_m-y)/(x_m-x))*(x_m-Step-x) + y ) ]
            return(good,x_m,y_m,ans,intersect)
        else:
            return(False,-1,-1,[],intersect)
    return(False,-1,-1,[],None)
    

def RRT_full_path(start, end, obstacles, mu, tolerance, ep6_plan, removed_pixels=None):
    goalNodes = [] # Approximate points for goal
    gx_min = max(end[0]-tolerance,XMIN1)
    gx_max = min(end[0]+tolerance, XMAX1)
    gy_min = max(end[1]-tolerance, YMIN1)
    gy_max = min(end[1]+tolerance, YMAX1)
    for gx in range(gx_min, gx_max):
        for gy in range(gy_min, gy_max):
            goalNodes.append(list((gx,gy)))
    print('Calculated goal region')
                    
    parent=dict()
    parent[start]=(-1,-1)
    Trace=[]
    Timer =  time.time()
    running = True
    Step = 10
    all_ans=[]
    xprev, yprev = start
    check_int = []
    while(running):
        temp = random_point((xprev,yprev), (mu,1-mu), (14,20), (5,5), removed_pixels)
        x,y=temp
        if (time.time() - Timer) > 5:
            Step=5
            
        good,x_m,y_m,ans,intersect=RRT(x,y,parent,obstacles,Step,ep6_plan)
        
        if ans != [] and ans[0] > XMAX1:
            ans[0] = XMAX1-1
        if ans != [] and ans[0] < XMIN1:
            ans[0] = XMIN1
        if ans != [] and ans[1] > YMAX1:
            ans[1] = YMAX1-1
        if ans != [] and ans[1] < YMIN1:
            ans[1] = YMIN1
        all_ans.append((ans))
            
        if good and ans:
            x_cur = ans[0]
            y_cur = ans[1]
            if (x_cur,y_cur) not in obstacles and (x_cur,y_cur) not in parent:
                if pass_thru_obstacle(x_cur, y_cur, x_m, y_m, ep6_plan) == False:
                    parent[(x_cur,y_cur)]=(x_m,y_m)
                    xprev, yprev = (x_cur, y_cur)
                    if [x_cur,y_cur] in goalNodes:
                        Trace=(x_cur,y_cur)
                        running = False
    print('Finished RRT')
    
    running = True
    PATH = []
    count = 0
    #This loop gets the route back to Start point
    while(Trace and running):
        while(Trace!=start):
            x,y = parent[Trace]
            Trace=(x,y)
            PATH.append((x,y))
            xprev,yprev = x,y
            if count != 0:
                inter = pass_thru_obstacle(x,y,xprev,yprev,ep6_plan)
                check_int.append(inter)
            count += 1
            
        if Trace==start:
            running=False

    print('Finished finding path')
    return PATH, check_int
