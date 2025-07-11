
import numpy as np 
import pandas as pd 

class NeuronIF:
    def __init__(self, threshold = 0.5, membrane_reset = 0, time_step = 0.5) -> None:
        self.threshold = threshold 
        self.membrane_reset = membrane_reset 
        self.time_step = time_step
        self.membrane_potential = 0.0 
        self.spiked = False 
        pass
    
    def update(self, input_current):
        # update membrane potential based on input current 
        self.membrane_potential += input_current * self.time_step 
        
        # check if the neuron has spiked 
        if self.membrane_potential >= self.threshold:
            self.spiked = True 
            self.membrane_potential = self.membrane_reset
        else:
            self.spiked = False 
            
        return self.spiked  
