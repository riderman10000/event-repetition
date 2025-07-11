import cv2 
import time 
import numpy as np 
import pandas as pd 
import torch
import matplotlib.pyplot as plt 

try:
    from . import tools as tl
except:
    import tools as tl 



def save_event_as_video(events, name,
    event_points = None, width = 128, height = 128,
    threshold = 0.9, decay = 0.005, image_scale = 2,
    frame_per_second = 1/60 ): # frame per second 
    # create image 
    frame_count = 0
    image = np.zeros((height, width, 3), dtype=np.uint16)
    
    if event_points:
        events = events[:event_points]
    
    prev_x, prev_y, prev_time = 0, 0, 0 
    
    peak, spike = 255, 255 
    
    video_writer = cv2.VideoWriter(
        name,
        # apiPreference=cv2.CAP_FFMPEG,
        cv2.VideoWriter_fourcc (*'MJPG'), # (*'H264'), # (*'XVID'), #
        1/frame_per_second, (width, height)
    )
    
    for event_idx, event in enumerate(events):
        time_stamp, x, y = event[0], int(event[1]), int(event[2])
        
        if not tl.is_with_in_manhattan_distance(prev_x, prev_y, x, y):
            prev_x, prev_y = x, y 
            continue
        prev_x, prev_y = x, y 
        
        # spike of the neuron 
        image[y, x] = spike 
        
        # decay of the neuron
        image = image - (decay * image ) #* (time_stamp - prev_time))
        image[image < 0] = 0 
        
        # print(event_idx, image.max())
        image_display = image.astype(np.uint8)
        
        cv2.imshow('test', cv2.resize(image_display, (width * image_scale, height * image_scale)))
        
        if ((time_stamp - prev_time) >  (frame_per_second * 1000000)):
            frame_count += 1
            video_writer.write(image_display) 
            print(f"[frame count: {frame_count} ] --- [time stamp] -- timestamp {time_stamp}  -- prevstamp -- {prev_time} -- diff {((time_stamp - prev_time))}")
            prev_time = time_stamp 
        
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
        
        ...
    print(time_stamp, prev_time)

    cv2.destroyAllWindows()
    video_writer.release() 

import torch
import time
import matplotlib.pyplot as plt

def get_event_frames_vectorized_mod(events, width=128, height=128, decay=0.0009, 
                                frame_interval_us=8333, device='cuda', visualize=False):
    """
    Vectorized version of get_event_frame with exponential decay between frames.
    
    Parameters:
        events: torch.Tensor of shape [N, 3] (timestamp [µs], x, y)
        frame_interval_us: Microseconds between frames (e.g., 1/60s = 16666 µs)
    """
    timestamps = events[:, 0]
    xs = events[:, 1].long()
    ys = events[:, 2].long()
    delta = 20

    # applying manhattan distance 
    diffs = torch.abs(torch.diff(xs)).to(device)
    mask1 = torch.cat([torch.tensor([True], device=device), diffs <=delta ])  # Add a zero for the first event      

    diffs = torch.abs(torch.diff(ys)).to(device)
    mask2 = torch.cat([torch.tensor([True], device=device), diffs <=delta ])  # Add a zero for the first event      
    mask = mask1 & mask2 

    timestamps = timestamps[mask]
    xs = xs[mask]
    ys = ys[mask]
    # Normalize timestamps and compute frame indices


    min_time = timestamps[0]
    relative_times = timestamps - min_time
    frame_indices = (relative_times // frame_interval_us).long()
    num_frames = int(frame_indices.max().item()) + 1

    frames = torch.zeros((num_frames, height, width), device=device)
    decay_factor = torch.exp(torch.tensor(-decay, device=device))  # Scalar decay

    # Add spikes
    frames.index_put_((frame_indices, ys, xs), torch.ones(len(frame_indices), device=device), accumulate=True)

    # Decay and accumulate over time
    for i in range(1, num_frames):
        frames[i] = frames[i] - frames[i - 1] * decay_factor
        frames[i][frames[i] < 0] = 0 
        # frames[i] = torch.clip(frames[i], 0, 1.0)

        if visualize:
            plt.imshow(frames[i].cpu().numpy(), cmap='gray')
            plt.title(f"Frame {i}")
            plt.pause(0.1)

    frames = torch.clip(frames, 0, 1.0)  # binary spike image

    return frames

def get_event_frames_vectorized(events, width=128, height=128, decay=0.009, 
                                frame_interval_us=8333, device='cuda'):
    """
    Vectorized version of get_event_frame.
    
    Parameters:
        events: torch.Tensor of shape [N, 3] (timestamp [µs], x, y)
        frame_interval_us: Microseconds between frames (e.g., 1/60s = 16666 µs)
    """
    # Unpack events
    timestamps = events[:, 0]
    xs = events[:, 1].long()
    ys = events[:, 2].long()

    # Normalize timestamps and compute frame indices
    min_time = timestamps[0]
    relative_times = timestamps - min_time
    frame_indices = (relative_times // frame_interval_us).long()

    num_frames = int(frame_indices.max().item()) + 1

    # Create empty tensor to hold frames [num_frames, H, W]
    frames = torch.zeros((num_frames, height, width), device=device)

    # # Count spikes per (frame, y, x)
    # indices = torch.stack([frame_indices, ys, xs], dim=1)  # [N, 3]
    # frames.index_put_(tuple(indices.T), torch.ones(len(events), device=device), accumulate=True)

    frames.index_put_((frame_indices, ys, xs), torch.ones(len(frame_indices), device=device), accumulate=True)

    
    # Apply decay (optional: simple normalization)
    frames = torch.clip(frames, 0, 1.0)  # binary spike image
    return frames

def get_event_frames_decay(events, width=128, height=128, decay=0.01, 
                           frame_interval_us=8333, device='cuda'):
    timestamps = events[:, 0]
    xs = events[:, 1].long()
    ys = events[:, 2].long()
    
    min_time = timestamps[0]
    relative_times = timestamps - min_time
    frame_indices = (relative_times // frame_interval_us).long()
    num_frames = int(frame_indices.max().item()) + 1

    frames = torch.zeros((num_frames, height, width), device=device)
    frame_buffer = torch.zeros((height, width), device=device)

    last_frame_idx = -1
    for i in range(events.shape[0]):
        t_idx = frame_indices[i].item()
        x, y = xs[i].item(), ys[i].item()

        # Apply decay to frame buffer if new frame
        if t_idx != last_frame_idx and last_frame_idx >= 0:
            decay_steps = t_idx - last_frame_idx
            # frame_buffer *= torch.exp(-decay * decay_steps)
            frame_buffer *= torch.exp(torch.tensor(-decay * decay_steps, device=frame_buffer.device))


        # Add spike
        frame_buffer[y, x] = 1.0

        # Save current frame
        frames[t_idx] = frame_buffer.clone()
        last_frame_idx = t_idx

    return torch.clip(frames, 0, 1.0)


def get_event_frame(events,
    event_points = None, width = 128, height = 128,
    threshold = 0.9, decay = 0.005, image_scale = 2,
    frame_per_second = 1/60 , event_time_include = False): # seconds per frame 
    image = np.zeros((height, width), dtype=np.uint8)
    
    if event_points:
        events = events[:event_points]

    prev_x, prev_y, prev_time = 0, 0, 0 
    peak, spike = 1, 1 # 255, 255 
    
    for event_idx, event in enumerate(events):
        time_stamp, x, y = event[0], int(event[1]), int(event[2])
        
        if not tl.is_with_in_manhattan_distance(prev_x, prev_y, x, y):
            prev_x, prev_y = x, y 
            continue
        prev_x, prev_y = x, y 

        # spike of the neuron 
        image[y, x] = spike

        # decay of the neuron
        image = image - (decay * image ) #* (time_stamp - prev_time))
        image[image < 0] = 0 

        if ((time_stamp - prev_time) >  (frame_per_second * 1000000)):
            yield image 

            # if event_time_include:
            #     print(f"[time stamp] -- timestamp {time_stamp}  -- prevstamp -- {prev_time} -- diff {((time_stamp - prev_time)/1000)}")
            #     time.sleep((time_stamp - prev_time)/1000)
            prev_time = time_stamp 
    # print(f"[time stamp] -- timestamp {time_stamp}  -- prevstamp -- {prev_time} -- diff {((time_stamp - prev_time)/1000)}")
        


if __name__ == "__main__":
    
    # load event camera data 
    events = tl.load_event_data('./event_csv/split_data/class2/user02_lab.csv')  # Replace with your actual file path
    save_event_as_video(events, 'test.mp4', frame_per_second=1/60)