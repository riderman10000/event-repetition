import torch
import numpy as np
import time
import glob 

from repnet.model import RepNet
import event_tools.tools as tl
import event_tools.e2v as e2v

# ----------------------------- Configs -------------------------------- #
event_class = "class2"
event_user = "user02_lab"
event_csv_file_name = f"./event_csv/split_data/{event_class}/{event_user}.csv"

weights_path =  './ssn.pth' # './pytorch_weights.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
strides = [1, 2, 3, 4, 8]

width = height = 128
fps = 120
frame_interval = 1 / fps
decay = 0.009
threshold = 0.8
spike = 1

# ----------------------------- Load Model ----------------------------- #
def load_model(path, device):
    model = RepNet(encoder=False)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model.to(device)

# ------------------------ Spiking Encoder ---------------------------- #
def process_events_to_spikes(events, decay, frame_interval, width=128, height=128, device='cuda', spike=1.0, threshold=0.8):
    horiz_potential = torch.zeros((1, width), device=device)
    vert_potential = torch.zeros((1, height), device=device)
    spikes = []

    # Predefined fixed weights
    w_horiz = torch.full((1, height), 0.5, dtype=torch.float32, device=device)
    w_vert = torch.full((1, height), 0.5, dtype=torch.float32, device=device)

    for frame in e2v.get_event_frame(events, frame_per_second=frame_interval, decay=decay):
        frame_tensor = torch.as_tensor(frame, dtype=torch.float32, device=device)

        horiz_potential.mul_(decay).add_(w_horiz @ frame_tensor)
        vert_potential.mul_(decay).add_(w_vert @ frame_tensor.T)

        spike_h = horiz_potential > (spike * threshold)
        spike_v = vert_potential > (spike * threshold)

        spikes.append(torch.cat([spike_h.float(), spike_v.float()], dim=-1))

        horiz_potential.masked_fill_(spike_h, 0).relu_()
        vert_potential.masked_fill_(spike_v, 0).relu_()

    return spikes


def encode_spikes_from_frames(frames, decay = 0.009, threshold=0.8, spike=1.0):
    N, H, W = frames.shape
    device = frames.device
    
    horiz_potential = torch.zeros((1, W), device=device)
    vert_potential = torch.zeros((1, H), device=device)
    
    # Fixed weights (same as before)
    w_horiz = torch.full((1, H), 0.5, dtype=torch.float32, device=device)
    w_vert = torch.full((1, H), 0.5, dtype=torch.float32, device=device)

    spikes = []

    for frame in frames:
        frame_tensor = frame.float()

        # Update membrane potentials
        horiz_potential.mul_(decay).add_(w_horiz @ frame_tensor)
        vert_potential.mul_(decay).add_(w_vert @ frame_tensor.T)

        # Thresholding to generate spikes
        spike_h = horiz_potential > (spike * threshold)
        spike_v = vert_potential > (spike * threshold)

        # Concatenate spikes
        spikes.append(torch.cat([spike_h.float(), spike_v.float()], dim=-1))

        # Reset neurons that spiked
        horiz_potential.masked_fill_(spike_h, 0).relu_()
        vert_potential.masked_fill_(spike_v, 0).relu_()

    return torch.stack(spikes)  # [N, H+W]


# ----------------------------- Inference ----------------------------- #
def run_stride_inference(spikes, model, stride):
    stride_frames = spikes[::stride]
    stride_frames = stride_frames[:(len(stride_frames) // 64) * 64]
    if len(stride_frames) < 64:
        return None

    period_lengths, periodicities, embeddings = [], [], []

    for i in range(0, len(stride_frames) - 63, 64):
        # chunk = torch.stack(stride_frames[i:i+64]).view(-1, 64, 256).to(device)
        chunk = stride_frames[i:i+64].view(-1, 64, 256).to(device)
        with torch.no_grad():
            p_len, p_score, emb = model(chunk)
            period_lengths.append(p_len[0].cpu())
            periodicities.append(p_score[0].cpu())
            embeddings.append(emb[0].cpu())

    period_lengths = torch.cat(period_lengths)
    periodicities = torch.cat(periodicities)
    embeddings = torch.cat(embeddings)

    confidence, period_length, period_count, periodicity_score = model.get_counts(
        period_lengths, periodicities, stride
    )
    return confidence, period_length, period_count, periodicity_score, embeddings

# ----------------------------- Main Logic ---------------------------- #
def main(model, event_class, event_user, file_path=None): 
    
    event_csv_file_name = f"./event_csv/split_data/{event_class}/{event_user}"
    if file_path:
        event_csv_file_name = file_path
    
    start_time = time.time()
    
    events = tl.load_event_data(event_csv_file_name)
    # print('time taken to load event data: {:.2f} seconds'.format(time.time() - start_time))
    # print("Processing spikes...")
    # spikes = process_events_to_spikes(events, decay, frame_interval)
    # events_np = np.array(events)  # shape [N, 3]
    events_tensor = torch.tensor(events, dtype=torch.float32, device='cuda')

    decay = 0.9 
    spikes = e2v.get_event_frames_vectorized_mod(
        events_tensor, width=128, height=128, visualize=0, decay=decay)
    # spikes = e2v.get_event_frames_decay(events_tensor)
    spikes = encode_spikes_from_frames(spikes, decay=decay)
    # time_interval = 1/60  # 10 ms per frame (time interval in microseconds)
    # decay = 0.009         # Decay factor for potential
    # spikes = e2v.process_events_to_spikes_by_time(events, decay, time_interval)
    
    # print('time taken to process spikes: {:.2f} seconds'.format(time.time() - start_time))
    best_result = None
    for stride in strides:
        result = run_stride_inference(spikes, model, stride)
        if result:
            confidence, period_length, period_count, periodicity_score, embeddings = result
            if not best_result or confidence > best_result[0]:
                best_result = (confidence, period_length, period_count, stride, periodicity_score, embeddings)

    if not best_result:
        raise RuntimeError("No valid 64-frame window found. Try smaller stride values.")

    confidence, period_length, period_count, best_stride, periodicity_score, _ = best_result
    # print(f"Predicted period length: {period_length / fps:.1f} sec (~{int(period_length)} frames), "
        #   f"confidence: {confidence:.2f}, using stride: {best_stride}")
    # print(f"Predicted period count: {round(period_count.tolist()[-1])}")
    print(f"{event_class}, {event_user}, {round(period_count.tolist()[-1])}")
    # print("Total time taken: {:.2f} seconds".format(time.time() - start_time))

def test_event(event_class, event_user): 
    import matplotlib.pyplot as plt
    event_csv_file_name = f"./event_csv/split_data/{event_class}/{event_user}"
    events = tl.load_event_data(event_csv_file_name)
    frame_per_second = 1/120
    for idx, frame_image in enumerate(
        e2v.get_event_frame(events, frame_per_second = frame_per_second,
        decay=0.009, event_time_include=True)):
        
        plt.imshow(frame_image, cmap='gray')
        plt.title(f"Frame {idx}")
        plt.pause(0.1)
        
        

def test_event_1(event_class, event_user): 
    import matplotlib.pyplot as plt
    event_csv_file_name = f"./event_csv/split_data/{event_class}/{event_user}"
    events = tl.load_event_data(event_csv_file_name)
    events_tensor = torch.tensor(events, dtype=torch.float32, device='cuda')
    frame_per_second = 1/120
    for idx, frame_image in enumerate(
        e2v.get_event_frames_vectorized_mod(events_tensor, width=128, height=128, decay=0.9)):
        
        plt.imshow(frame_image.cpu().numpy(), cmap='gray')
        plt.title(f"Frame {idx}")
        plt.pause(0.1)


if __name__ == "__main__":
    # files = sorted(glob.glob(f"./event_csv/split_data/class6/*_natural.csv"))
    files = sorted(glob.glob(f"./event_csv/split_data/class6/*.csv"))
    
    # file_path = "/home/rlwagun/Files/action_count_in_events3/all_code/time_series/artificial_data/individual/class2/*.csv"
    # file_path = "/home/rlwagun/Files/action_count_in_events3/all_code/time_series/artificial_data/combined_same_class/*.csv"
    # files = sorted(glob.glob(file_path))
    
    model = load_model(weights_path, device)
    for file in files:
        event_class = file.split('/')[-2]
        event_user = file.split('/')[-1]
        # inference(model, class_name, user_name)
        main(model, event_class, event_user, file)
        # test_event(event_class, event_user)
        # test_event_1(event_class, event_user)