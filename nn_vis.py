import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from snn import NeuronIF
import time

class NN_Graph:
    def __init__(self, ax, left, right, bottom, top, layer_sizes) -> None:
        '''
        Draw a neural network cartoon using matplotlib.
        '''
        self.circles = []
        self.edges = [] 
        self.ax = ax 
        
        n_layers = len(layer_sizes)
        v_spacing = (top - bottom)/float(max(layer_sizes))
        v_spacing += .005
        h_spacing = (right - left)/float(len(layer_sizes) - 1) + 2 if (len(layer_sizes) - 1) != 0 else 0 

        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
            layer_circle = []
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/2.,
                                    color='w', ec='k', zorder=4)
                layer_circle.append(circle)
                self.ax.add_artist(circle)
            self.circles.append(layer_circle)

        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                    [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                    self.ax.add_artist(line)

    def update(self, layer=0, neurons_values=None):
        if neurons_values is None:
            neurons_values = np.random.rand(128)
        for neuron_index, neuron_value in enumerate(neurons_values):
            self.update_neuron(layer, neuron_index, neuron_value)

    def update_neuron(self, layer=0, neuron_index=0, neuron_value=0):
        self.circles[layer][neuron_index].set_color([neuron_value, 0, 0])

# Initialize the figure and axis
# fig = plt.figure(figsize=(15, 15))
# ax = fig.gca()
fig, (ax, ax1) = plt.subplots(1, 2)


ax.axis('off')

# Create the neural network graph
nn_graph = NN_Graph(ax, .05, .9, .1, .9, [128])

# Define the first layer number of neurons 
first_layer_number_of_neurons = 128
first_layer_neurons = [NeuronIF() for _ in range(first_layer_number_of_neurons)]

# Import data (assuming it's a CSV with x and y columns)
file_path = './user02_lab.csv'
data_df = pd.read_csv(file_path)
data_df = data_df[['x', 'y']]

ax1 = plt.scatter(x =[_ for _ in range(len(data_df))], y= data_df['y'], 
    s=1)

# Set up the counter for animation
counter = 0

# Define the update function for the animation
def update(frame):
    global counter

    if counter >= len(data_df):  # Stop if we reach the end of the data
        print('final _counter ', counter)
        return 

    # Update neuron state based on the data
    neuron_index = int(data_df.iloc[counter]['x'])
    first_layer_neurons[neuron_index].update(0.25)  # Update neuron with some value

    # Update the neural network visualization
    nn_graph.update_neuron(
        layer=0, 
        neuron_index=neuron_index,
        neuron_value=first_layer_neurons[neuron_index].membrane_potential
    )
    
    circle = plt.Circle((counter, data_df.iloc[counter]['x']), 1,
                                    color='r', ec='k', zorder=4)
    # layer_circle.append(circle)
    # ax1.add_artist(circle)
    # ax1.update()
    # # Increment the counter
    ax.add_patch(circle)
    counter += 1
    print('counter ', counter)
    

    return nn_graph.circles[0]  # Return the list of circles to update in animation

# Create the animation
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=100,  # Set the number of frames
    interval=200,  # Time interval between frames (in ms)
    blit=True  # Only update the objects that changed (optimized)
)

plt.show()
