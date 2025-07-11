import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2

import event_tools.tools as tl 

# Example function to load event camera data (This depends on the format you're using)
def load_event_data(file_path):
    # Placeholder function to load event data from a file
    # In real code, you would read the file and return events as an array with [timestamp, x, y, polarity]
    # For example:
    # events = np.loadtxt(file_path) 
    
    data_df = pd.read_csv(file_path)
    data_df = data_df[['x', 'y']]
    
    events = np.random.rand(len(data_df), 3)  # For the sake of illustration, generate random events
    events[:, 0] = [_ for _ in range(len(data_df))]  # np.floor(events[:, 0] * 1000)  # Random timestamps
    events[:, 1] = data_df['x'].to_numpy(dtype=np.int32) # np.floor(events[:, 1] * 640)  # Random x-coordinates (640x480 resolution)
    events[:, 2] = data_df['y'].to_numpy(dtype=np.int32) # np.floor(events[:, 2] * 480)  # Random y-coordinates (640x480 resolution)
    return events

# Function to visualize events as a scatter plot (spatial distribution)
def visualize_events_scatter(events):
    plt.figure(figsize=(10, 6))
    plt.scatter(events[:, 1], events[:, 2], c=events[:, 0], cmap='jet', s=1, alpha=0.5)
    plt.title("Event Camera Data (Scatter View)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.colorbar(label="Timestamp")
    plt.show()

# Function to generate an intensity image from events (accumulate events over time)
def generate_intensity_image(events, width=640, height=480, time_window=5000):
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

# Function to calculate and display the grayscale histogram
def display_grayscale_histogram(image):
    # Calculate the histogram of the grayscale image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Normalize the histogram (optional)
    hist /= hist.sum()

    # Create an empty black image to display the histogram
    hist_image = np.zeros((300, 256, 3), dtype=np.uint8)

    # Normalize the histogram for scaling the values
    hist = hist * hist_image.shape[0]
    
    # Draw the histogram on the empty image
    for i in range(1, 256):
        cv2.line(hist_image, (i-1, int(hist[i-1])), (i, int(hist[i])), (255, 255, 255), 2)

    # Display the histogram using OpenCV
    cv2.imshow('Grayscale Histogram', hist_image)


# Function to visualize events in a more interactive way (e.g., using OpenCV)
def visualize_events_opencv(events, width=128, height=128, time_window=1000):
    # Create a black image
    img = np.zeros((height, width), dtype=np.uint32)
    
    column_events = np.zeros(width)
    row_events = np.zeros(height)
    
    # Loop through events and draw them
    
    prev_x = 0 
    prev_y = 0 
    
    decay = .1
    
    horizontal_neurons = np.zeros_like(img[0, :])
    vertical_neurons = np.zeros_like(img[:, 0])
    
    for event in events:
        time_stamp = event[0]
        print(f"event time: {time_stamp}")
        
        x, y = int(event[1]), int(event[2])
        if not tl.is_with_in_manhattan_distance(prev_x, prev_y, x, y):
            prev_x, prev_y = x, y 
            continue
        prev_x, prev_y = x, y 
        
        # spike of the neuron
        img[y, x] = 255  # Set the pixel to white (event occurrence)
        
        # decay of the neuron
        img = img - (decay * img )
        img[img < 0] = 0 
        horizontal_fig = img.sum(axis=0)
        # img[0, :] = horizontal_fig
        vertical_fig = img.sum(axis=1)
        # img[:, 0] = vertical_fig

        
        column_events[x] = 255
        column_events = column_events - (decay * column_events)
        column_events[column_events < 0] = 0
        img[1, :] = column_events 
        
        row_events[y] = 255
        row_events = row_events - (decay * row_events)
        row_events[row_events < 0] = 0
        img[:, 1] = row_events 
                
        # Display image using OpenCV
        cv2.imshow('Event Camera Visualization', cv2.resize(img, (width*2, height*2) ))
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        # if time_stamp%time_window == 0:
        #     img = np.zeros((height, width), dtype=np.uint8)
    cv2.destroyAllWindows()

# Main function to demonstrate visualization
def main():
    # Load event camera data
    events = load_event_data('./event_csv/split_data/class7/user02_lab.csv')  # Replace with your actual file path

    # Visualize using scatter plot (spatial distribution)
    visualize_events_scatter(events)

    # Visualize as an intensity image over time
    generate_intensity_image(events, time_window=5000)  # Example time window in ms

    # Visualize in OpenCV (real-time display)
    visualize_events_opencv(events)

if __name__ == "__main__":
    main()
