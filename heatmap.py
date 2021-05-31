# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 01:48:27 2021

@author: chait
"""

from PIL import Image
from copy import deepcopy
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    # https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def color_scales(c1, c2, n):
    scale=[]
    for i in range(n):
        scale.append(colorFader(c1,c2,i/n))
    return scale


def hex_to_rgb(hex_val,alpha):
    hex_val = hex_val.lstrip('#')
    rgb_val = list(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
    rgb_val.append(alpha)
    return rgb_val


def view_color_scales(c1, c2, n):
    # Use: To view how the color scales chosen look
    
    # Inputs:
    # c1, c2 - String hex values of colors, eg: '#ff0000'. 
    # Get color values from: https://www.101computing.net/rgb-converter/
    # n - Number of colors wanted in between c1 and c2, i.e controls fineness
    
    # Outputs:
    # Does not return anything. 
    # Displays a plot containing all vertical lines of colors in range selected above
    fig, ax = plt.subplots(figsize=(8, 5))
    for x in range(n+1):
        ax.axvline(x, color=colorFader(c1,c2,x/n), linewidth=4) 
    plt.show()
    
    
def dynamic_heatmap_model2(risk_matrix_dynamic, pos, ep6_plan):
    
    heatmap = []
    heatmap_with_people = []
    
    ep6_plan = Image.fromarray(ep6_plan)
    ep6_plan = ep6_plan.convert("RGBA")
    
    n = 100 # Number of color gradient lines 
    scale_red_orange = color_scales('#ff0000', '#ff8100', n)
    scale_orange_yellow = color_scales('#ff8100', '#ffcd00', n)
    scale_yellow_green = color_scales('#ffcd00', '#55af00', n)
                                      
    # Make array of color all values
    risk_hex_scale = np.concatenate((scale_red_orange, scale_orange_yellow, scale_yellow_green))
    risk_rgb_scale = []
    for i in range(len(risk_hex_scale)):
        risk_rgb_scale.append(hex_to_rgb(risk_hex_scale[i], alpha=100))
    risk_rgb_scale = list(reversed(risk_rgb_scale))
    
    for i in range(len(risk_matrix_dynamic)):
        print('Processing frame ', i, '/', len(risk_matrix_dynamic))
        risk_frame = 255*np.ones((np.shape(ep6_plan)[0], np.shape(ep6_plan)[1], 4))
        risk_mat = np.tanh(risk_matrix_dynamic[i])
        pixels_to_color = deepcopy(risk_mat)
        risk_mat = ((n-1)*risk_mat).astype(int)
        #risk_mat = (risk_mat).astype(int)
        
        for j in range(np.shape(risk_mat)[0]):
            for k in range(np.shape(risk_mat)[1]):
                if risk_mat[j,k] >= 3*n:
                    if pixels_to_color[j,k] > 0:
                        risk_frame[j,k] = np.array(risk_rgb_scale[-1]).astype(int)
                else:
                    if pixels_to_color[j,k] > 0.00001: #0.005:
                        risk_frame[j,k] = np.array(risk_rgb_scale[risk_mat[j,k]]).astype(int)
                        
        risk_frame_ppl = deepcopy(risk_frame)
        for j in range(len(pos[i])):
            risk_frame_ppl[pos[i][j][0]-2:pos[i][j][0]+2, pos[i][j][1]-2:pos[i][j][1]+2, :] = [0,0,0,255]
        
        # Superimpose on map
        risk_frame = Image.fromarray(risk_frame.astype(np.uint8))
        risk_frame = risk_frame.convert("RGBA")
        risk_frame = Image.blend(risk_frame, ep6_plan, 0.3)   
        
        risk_frame_ppl = Image.fromarray(risk_frame_ppl.astype(np.uint8))
        risk_frame_ppl = risk_frame_ppl.convert("RGBA")
        risk_frame_ppl = Image.blend(risk_frame_ppl, ep6_plan, 0.3)   

        heatmap.append(risk_frame)
        heatmap_with_people.append(risk_frame_ppl)
        
    return heatmap, heatmap_with_people
