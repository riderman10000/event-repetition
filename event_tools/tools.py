import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def compute_distance(x1,y1,x2,y2):
    return abs(x1-x2),abs(y1-y2)

def is_with_in_manhattan_distance(prev_x, perv_y, current_x, current_y, delta=5,count_margin=100,nature_flag=True): 
    x_dis, y_dis = compute_distance(prev_x, perv_y, current_x, current_y)
    # focus on up to count_margin a point 
    if x_dis <= delta and y_dis <= delta:
        return True
    
# Example function to load event camera data (This depends on the format you're using)
def load_event_data(file_path):
    # Placeholder function to load event data from a file
    # In real code, you would read the file and return events as an array with [timestamp, x, y, polarity]
    # For example:
    # events = np.loadtxt(file_path) 
    
    data_df = pd.read_csv(file_path)
    # data_df = data_df[['x', 'y']]
    # print(data_df)
    events = np.random.rand(len(data_df), 3)  # For the sake of illustration, generate random events
    # events[:, 0] = [_ for _ in range(len(data_df))]  # np.floor(events[:, 0] * 1000)  # Random timestamps
    events[:, 0] = data_df['timestamp'].to_numpy() # [_ for _ in range(len(data_df))]
    events[:, 1] = data_df['x'].to_numpy(dtype=np.int32) # np.floor(events[:, 1] * 640)  # Random x-coordinates (640x480 resolution)
    events[:, 2] = data_df['y'].to_numpy(dtype=np.int32) # np.floor(events[:, 2] * 480)  # Random y-coordinates (640x480 resolution)
    
    # exit()
    return events



# Function to visualize events as a scatter plot (spatial distribution)
def visualize_events_scatter(events):
    """
    function to visualize events as a scatter plot (spatial distribution)
    """
    
    plt.figure(figsize=(10, 6))
    plt.scatter(events[:, 1], events[:, 2], c=events[:, 0], cmap='jet', s=1, alpha=0.5)
    plt.title("Event Camera Data (Scatter View)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.colorbar(label="Timestamp")
    plt.show()

# Function to generate an intensity image from events (accumulate events over time)
def generate_intensity_image(events, width=640, height=480, time_window=5000):
    """
    function to generate an intensity image from events (accumulate events over time)
    """
    
    # Create a blank frame (image) to accumulate events
    intensity_image = np.zeros((height, width), dtype=np.int32)

    # Accumulate events within a given time window
    for event in events:
        timestamp, x, y = event
        x, y = int(x), int(y)
        if timestamp < time_window:  # Filter events within the time window
            print(y, x)
            intensity_image[y, x] += 1  # Increment intensity at the pixel location

    # Normalize the intensity image to display
    intensity_image = np.clip(intensity_image, 0, 255).astype(np.uint8)

    # Display the image
    plt.imshow(intensity_image, cmap='gray')
    plt.title(f"Intensity Image (Time Window: {time_window} ms)")
    plt.show()
