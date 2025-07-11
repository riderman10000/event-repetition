import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np 
import pandas as pd 

class NeuronIF:
    def __init__(self, threshold = 0.1, membrane_reset = -.5, time_step = 0.1) -> None:
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

# define the first layer number of neurons 
first_layer_number_of_neurons = 128
first_layer_neurons = [NeuronIF() for _ in range(first_layer_number_of_neurons)]

file_path = './user02_lab.csv'
data_df = pd.read_csv(file_path)
data_df = data_df[['x', 'y']]

data_df.head(), type(data_df), data_df['x'][0] 


# Generate some random data for the heatmap
data = np.array([neuron.membrane_potential for neuron in first_layer_neurons])
data = np.reshape(data, (1, -1))
data = np.concatenate([data]*4)

fig, ax = plt.subplots()

# Initialize the heatmap
im = ax.imshow(data, cmap='hot', interpolation='nearest')
counter = 0 
# Function to update the heatmap for each frame
def update(frame):
    global counter
    # data = np.random.rand(10, 10)
    # im.set_data(data)
    # return [im]
    print(f'[x] {counter}')
    if (counter > data_df.shape[0]) or  (counter == 100):
        exit()
    first_layer_neurons[data_df.iloc[counter]['x']].update(1)
    data = np.array([neuron.membrane_potential for neuron in first_layer_neurons])
    data = np.reshape(data, (1, -1))
    im.set_data(data)
    counter += 1 
    return [im]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=200, blit=True)

# Show the animation
plt.show()
ani.save(filename='heatmap_animation.gif', writer='pillow')