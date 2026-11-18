# Design goal

This task develops a reinforcement-learning (RL)-based lane-changing model that makes safe, efficient lane-changing decisions and executes maneuvers in large-scale traffic simulations.

# Model requirements

The mandatory requirements are:

* Executable on a graphics processing unit (GPU).  
* Use an RL model to perform lane changes under various system states.  
* Capable of making the following lane-changing decisions, and canceling them if the traffic conditions no longer allow these lane changes.  
  * Discretionary lane changes to bypass a slow-moving vehicle, even when the lane change may make the ego vehicle temporarily leave its desired lane;  
  * Discretionary lane changes that lead the ego vehicle to join a shorter queue at intersections or congested freeways;  
  * Proactive lane changes toward the turning lane before the next turn at an intersection or toward the ramp lane before the next on/off-ramp on freeways;  
  * Proactive lane changes to create a gap for other vehicles to enter the ego vehicle’s current lane. For example, change to the center lane to allow an on-ramp vehicle to enter the freeway;  
  * Mandatory lane changes that enable the ego vehicle to cut into a desired lane;  
* Capable of finding gaps for lane changes:  
  * Capable of finding safe gaps in the target lane.  
  * Capable of speeding up or slowing down in the current lane to search for a gap in the target lane if no gap is available.   
* Capable of implementing safe and efficient lane-changing maneuvers:  
  * Minimize negative impacts on the lag vehicle in the target lane.  
  * Optimize control actions to achieve desired lane changes.  
  * Avoid collisions during lane changes.  
* Capable of helping other lane-changing vehicles:  
  * Capable of slowing down in the current lane to safely accommodate cut-in vehicles from adjacent lanes.  
  * Capable of speeding up in the current lane to create a gap for vehicles that want to change to the ego vehicle’s lane.

Here are optional requirements:

* The model can depict different user aggressiveness levels.  
* The model can depict different information levels. For example, at a low information level, the model can search gaps within a short range (e.g., 50 meters). At a high information level, the model can search gaps at a long range (e.g., 200 meters).   
* The model structure should be as simple as possible to reduce the computation load. 

The following functions are not required:

* High-level trajectory planning. If the ego vehicle cannot make a desired lane change and misses a turn as a result, a separate trajectory planning algorithm will plan a new route for the ego vehicle. The lane-changing model does not need to plan the ego vehicle's route.

# Model training

We need to define a model structure and design a model training plan. The starting point is the GPU-accelerated simulation tool LPSim (Large-scale Parallel Simulator) here: [https\://github.com/Xuan-1998/LPSim](https://github.com/Xuan-1998/LPSim). 

# Model testing

The trained model should be tested under the following scenarios. We need to develop model evaluation metrics.

* Freeway with on- and off-ramps;  
* Freeway with lane drops;  
* Freeway with lane additions;  
* Signalized intersections with dedicated turning lanes;  
* Signalized intersections with shared turning lanes;  
* Stop-sign controlled intersections;  
* Yield-sign controlled intersections;  
   