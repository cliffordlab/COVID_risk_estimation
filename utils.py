# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 00:59:10 2021

@author: chait
"""

import numpy as np
import random

############################################################ Used in simulation.py #################################################

def random_node(xmin, xmax, ymin, ymax, obstacles):
    # Use: Find random node within bounding box. Used to find specific random 
    # points within start and end regions chosen by user.
    
    # Inputs:
    # xmin - minimum x coordinate of starting region bounding box
    # xmax - maximum x coordinate of starting region bounding box
    # ymin - minimum y coordinate of starting region bounding box
    # ymax - maximum y coordinate of starting region bounding box
    # obstacles - List of (x,y) coordinates that are walls or boundaries
    
    # Outputs:
    # (x,y) which is random point within a bounding box
    
    x = random.randint(xmin, xmax)
    y = random.randint(ymin, ymax)
    while (x,y) in obstacles:
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
    return (x,y)


def points_inside_rect(rect_coords):
    # Use: To get all points within a bounding box. Used to get all points considered
    # blocked by user.
    
    # Inputs:
    # rect_coords - (x,y) position of one corner and distance to other corner
    # like [x1, y1, x_length, y_length]
    
    # Outputs:
    # List of all points within rectangle
    
    removed_points = []
    for i in range(len(rect_coords)):
        x1 = rect_coords[i][1]
        y1 = rect_coords[i][0]
        x2 = rect_coords[i][1] + rect_coords[i][3]
        y2 = rect_coords[i][0] + rect_coords[i][2]
        for x in range(min(x1,x2), max(x1,x2)):
            for y in range(min(y1,y2), max(y1,y2)):
                if (x,y) not in removed_points:
                    removed_points.append((x,y))
    return removed_points


def animate(i,vl,period):
    # Use: To animate box plots
    t = i*period / 1000
    vl.set_xdata([t,t])
    return vl,

def add_pause(person_path, pause_len, start_ind):
    # Add pause for person to stand in one place for a while
    for i in range(pause_len):
        person_path.insert(start_ind+1, person_path[start_ind])
    return person_path

######################################################### Used in rrt_paths.py #################################################
def createLineIterator(P1, P2, img):
    # Use: Create line between two point P1 and P2. Used in pass_thru_obstacle
    # Note: P1, P2 take values as (x,y) itself and not (y,x) as is the case everywhere else
    
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]
    
    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)
    
    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)
    
       #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = float(dX)/float(dY)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.float) + P1X
        else:
            slope = float(dY)/float(dX)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.float) + P1Y
    
       #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]
    
    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]
    
    return itbuffer

def pass_thru_obstacle(x,y,xprev,yprev,ep6_plan):
    # Use: Check if part of path crosses obstacle. Needed to ensure that even if 
    # randomly chosen (x,y) does not lie on obstacle, it also does not pass thru one while
    # traveling from previous point
    
    # Inputs:
    # x,y - coordinates of current position
    # xprev, yprev - coordinates of position at previous time step
    # ep6_plan - RGB image of plan
    
    # Outputs:
    # intersect - False if line through (x,y) and (xprev,yprev) does not pass through
    # obstacle. Else it is true.
    
    points = createLineIterator((y,x), (yprev,xprev), ep6_plan[:,:,0])
    intersect = False
    for i in range(len(points)):
        if points[i][2] <= 100: # 100 is threshold to consider pixel as black
            intersect = True
            break
    return intersect


#Function Definition : Point to Point Distance
def p2p_dist(p1,p2):
    # Use: Find distance between two points
    
    # Inputs:
    # p1, p2 - points as (x,y)
    
    # Outputs:
    # Euclidean distance between p1, p2
    
    x1,y1=p1
    x2,y2=p2
    return ( ( (x1-x2)**2 + (y1-y2)**2 )**0.5 )


############################################################# Used in sd_metric.py #############################################
XMIN1 = 12 #15
XMAX1 = 115 #111
YMIN1 = 24 #22
YMAX1 = 210 #198

def check_wall_between_ppl(p1, p2, ep6_plan):
    # Use: Check if there is a wall between p1 and p2
    
    # Inputs: 
    # p1, p2 - points to check if there is a wall between
    # ep6_plan - RGB plan image
    
    # Outputs:
    # intersect - True if there is wall between p1, p2. Else, false

    points = createLineIterator(p1, p2, ep6_plan[:,:,0])
    
    points = points[np.where(points[:,0] >= YMIN1)]
    points = points[np.where(points[:,0] < YMAX1)]
    points = points[np.where(points[:,1] >= XMIN1)]
    points = points[np.where(points[:,1] < XMAX1)]
    
    bool_array = points[:,2]>100
    if bool_array.all() == True:
        intersect = False # If all points lie on white background, no intersection with boundaries
    else:
        intersect = True
        
    return intersect
