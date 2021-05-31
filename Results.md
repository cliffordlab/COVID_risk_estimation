# Experiments and results from the paper
This page contains all the results for the experiments described in the paper <add link to paper>
  
## Experiment 1 results - Effect of number of people in space
Aim was to check how the social distancing metric and risk metric change with number of people in the room. We simulate N people walking towards each other startingfrom points along the circumference of a circle of radius 8m.They stop when they are 1m away from each other and returnto their original position.  Multiple such simulations are runwith different values of N. We used N = 2, 4, 10 and 30 peo-ple. This experiment shows how the social distancing metricand risk metric vary when number of people and distancesbetween them varies.

### N = 2

Simulation of 2 people in environment  | Heatmap 
:-------------------------------------:|:-----------:
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl2/pplWalking_2.gif)|![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl2/heatmap_2.gif)

Box plot of distances between people |
:-----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl2/distBoxPlot_2.gif)|

Box plot of social distancing metric |
:----------------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/results/metricTests/Exp1/ppl2/SDmetricBoxPlot_2.gif)

Box plot of risk of exposure metric |
:----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl2_oldR/RiskBoxPlot_2.gif)

<br/><br/>

### N = 4

Simulation of 4 people in environment  | Heatmap 
:-------------------------------------:|:-----------:
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl4/pplWalking_4.gif)|![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl4/heatmap_4.gif)

Box plot of distances between people | 
:-----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl4/distBoxPlot_4.gif)

Box plot of social distancing metric | 
:-----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl4/SDmetricBoxPlot_4.gif)

Box plot of risk of exposure metric |
:----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl4_oldR/RiskBoxPlot_4.gif)

<br/><br/>

### N = 10

Simulation of 10 people in environment  | Heatmap 
:-------------------------------------:|:-----------:
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl10/pplWalking_10.gif)|![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl10/heatmap_10.gif)

Box plot of distances between people |
:-----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl10/distBoxPlot_10.gif)

Box plot of social distancing metric | 
:-----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl10/SDmetricBoxPlot_10.gif)

Box plot of risk of exposure metric |
:----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl10_oldR/RiskBoxPlot_10.gif)

<br/><br/>

### N = 30 

Simulation of 30 people in environment  | Heatmap 
:-------------------------------------:|:-----------:
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl30/pplWalking_30.gif)|![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl30/heatmap_30.gif)

Box plot of distances between people | 
:-----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl30/distBoxPlot_30.gif)

Box plot of social distancing metric | 
:-----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl30/SDmetricBoxPlot_30.gif)

Box plot of risk of exposure metric |
:----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp1/ppl30_oldR/RiskBoxPlot_30.gif)

<br/><br/>


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Experiment 2 results - Effect of time spent in same location
The aim of this experiment is to show howthe social distancing and risk metrics vary when N people re-main in the same position for time T, for example when theyare sitting in a class room.   We set N=5 for all the experi-ments.  We place the 5 people along the circumference of acircle of radius D, where we vary D. We use D = 1m, 2m,6m and 10m.  The 5 people are at the same position for T =50 seconds. For each of the D’s we run separate experimentswhere the distance between people is varied.

### Radius = 1m

Simulation of 2 people in environment  | Heatmap 
:-------------------------------------:|:-----------:
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist1m/pplWalking_1m.png)|![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist1m/heatmap_1m.gif)

Box plot of risk of exposure metric |
:----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist1m_oldR/RiskBoxPlot_5.gif)

Average distance between people: 0.98m
Average social distancing metric value: 6.12

<br/><br/>

### Radius = 2m

Simulation of 2 people in environment  | Heatmap 
:-------------------------------------:|:-----------:
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist2m/pplWalking_2m.png)|![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist2m/heatmap_2m.gif)

Box plot of risk of exposure metric |
:----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist2m_oldR/RiskBoxPlot_5.gif)

Average distance between people: 1.9m
Average social distancing metric value: 1.6

<br/><br/>

### Radius = 6m

Simulation of 2 people in environment  | Heatmap 
:-------------------------------------:|:-----------:
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist6m/pplWalking_6m.png)|![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist6m/heatmap_6m.gif)

Box plot of risk of exposure metric |
:----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist6m_oldR/RiskBoxPlot_5.gif)

Average distance between people: 5.72m
Average social distancing metric value: 6.35e-5

<br/><br/>

### Radius = 10m

Simulation of 2 people in environment  | Heatmap 
:-------------------------------------:|:-----------:
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist10m/pplWalking_10m.png)|![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist10m/heatmap_10m.gif)

Box plot of risk of exposure metric |
:----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/metricTests/Exp2/dist10m_oldR/RiskBoxPlot_5.gif)

Average distance between people: 9.5m
Average social distancing metric: 7.46e-14

<br/><br/>

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Experiment 3 results - Effect of time spent in same location
In this experiment, we simulate people mov-ing in the EP6 setting.  We show how the social distancingand risk metrics change as people move around this space.We simulate 30 people moving between different rooms in the space.

Simulation of 2 people in environment  | Heatmap 
:-------------------------------------:|:-----------:
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/ep6Tests/pplWalking_30.gif)|![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/ep6Tests/heatmap_30.gif)

Box plot of distances between people |
:-----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/ep6Tests/distBoxPlot_30.gif)|

Box plot of social distancing metric |
:----------------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/results/ep6Tests/SDmetricBoxPlot_30.gif)

Box plot of risk of exposure metric |
:----------------------------------:|
![](https://github.com/cliffordlab/CEP/blob/master/social_distancing/ep6_production/results/ep6Tests/RiskBoxPlot_30.gif)
