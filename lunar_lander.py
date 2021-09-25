#!/usr/bin/env python

"""RandomWalk environment class for RL-Glue-py.
"""

from environment import BaseEnvironment
import numpy as np
import gym

class LunarLanderEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """
        self.env = gym.make("LunarLander-v2")
        

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """        
        
        reward = 0.0
        observation = self.env.reset()
        is_terminal = False
                
        self.reward_obs_term = (reward, observation, is_terminal)
        
        # return first state observation from the environment
        return self.reward_obs_term[1]
        
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        last_state = self.reward_obs_term[1]
        current_state, reward, is_terminal, _ = self.env.step(action)
        reward=0
        (pos_x, pos_y,vel_x,vel_y,angle,angular_vel,a,b)=current_state 

        (abs_pos_x, abs_pos_y,abs_vel_x,abs_vel_y,abs_angle,abs_angular_vel,a,b)=tuple(np.abs(current_state))    
            
        if not is_terminal:
            penalize_angle= -10 if (abs_angle>1.5) else 0
            penalize_position=-100*abs_pos_x
            penalize_y_velocity=-10 if (abs_vel_y>1) else 0
            penalize_ang_vel= -10 if (abs_angular_vel>3) else 0
            reward=(10/np.exp(abs_pos_y))+penalize_position+penalize_angle+penalize_y_velocity
           
        else:

            if(abs_vel_y>0.3):
                reward+=-1000
                
            if(abs_vel_x>0.3):
                reward+=-1000
                   
                   
            if(abs_angle>0.1):
                reward+=-100
                    
            if(abs_pos_x>0.12):
                reward+=-1000
            reward+=a*b*1000
        self.reward_obs_term = (reward, current_state, is_terminal)
        
        return self.reward_obs_term