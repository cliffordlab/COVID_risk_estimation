# Social Distancing and Risk Estimation Simulation

This project simulates people walking around a built environment and measures social distancing and estimates risk of exposure to COVID-19 in the environment.

Refer to this paper for more details <link to paper>
Please cite this work if you use it: <Add citation>

## Explanation of concept
The code has four main parts:
1) Simulation of people walking around a built environment
2) Social distancing metric 
3) Risk of exposure metric
4) Heatmap representation of 2 and 3 in the built environment

### 1) Simulation of people walking around built environment
User inputs the floor plan of the built environment. User can remove walls/obstacles or block off certain areas of the built environment.
User inputs total number of people in the simulation and interactively selects their starting and destination areas.
Rapidly exploring random trees are used to build random paths for the individuals in simulation from their respective start position to respective destination positions.
RRT is modified to make people walk in a Levy walk or Brownian motion depending on how healthy they are (w.r.t MCI). This is controlled by the factor mu which is an input fron user.

### 2) Social distancing metric
Social distancing metric is used to quantify how well social distancing is being practiced in the environment. A higher value indicates more crowding. The equation to
calculate the metric is as follows:

<img src="https://latex.codecogs.com/svg.image?\phi&space;=&space;\sum_{i,j;&space;i\neq&space;j}^{N}&space;e^{-(d_{ij}^2&space;-&space;\alpha^2)/\alpha^2}" title="\phi = \sum_{i,j; i\neq j}^{N} e^{-(d_{ij}^2 - \alpha^2)/\alpha^2}" />

where,

<img src="https://latex.codecogs.com/svg.image?\phi&space;" title="\phi " /> - social distancing metric 

<img src="https://latex.codecogs.com/svg.image?N&space;" title="N " /> - total number of people in the simulation 

<img src="https://latex.codecogs.com/svg.image?d_{ij}" title="d_{ij}" /> - distance between persons i and j 

<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" /> - recommended social distance to be maintained (eg. 6ft) 


### 3) Risk of exposure metric
The risk of exposure metric quantifies the risk of an individual to be exposed to COVID-19. It takes into account details such as person's age, mask wearing, susceptibility 
due to underlying conditions, rate of ventilation in the built environment, half life of the virus in environment, etc.

### 4) Heatmap representation of 2 and 3 in the built environment
The risk of exposure in the environment is visualized in the form of a dynamic heatmap. This helps with planning of setup of space and number of people
  allowed inside at a time.
  
  
## Running the code
Clone the repository.
Running **simulation.py** starts the interactive program. User input is necessary for the code to run.

**sd_metric** contains code for the social distancing metric and risk metric.

**rrt_paths** has code for path planning of individuals in the simulation.

**heatmap.py** has code to generate the heatmap.

**probability_distribution.py** has code to model Levy and Brownian walks.

**utils.py** contains miscelaneous code needed to run the code.


simulation.py calls the functions it requires from the scripts above.

