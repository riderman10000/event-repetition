import os 
import cv2 
import glob 
import argparse

import torch
import torchvision.transforms as T

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from repnet import utils, plots
from repnet.model import RepNet

import event_tools.tools as tl 
import visualization as vz
import event_tools.e2v as e2v


def main(event_class, event_user, 
        out_csv_file = "encoder_replaced_output.csv",  weights = './pytorch_weights.pth',
        device = 'cuda'):
    # event_csv_file_name = f"./event_csv/split_data/{event_class}/{event_user}.csv"
    event_csv_file_name = f"./event_csv/split_data/{event_class}/{event_user}.csv"

    count_data = {
        "class": event_class,
        "condition": event_user,
        "count": 0,
    }
    if os.path.exists(out_csv_file):
        try: 
            out_df = pd.read_csv(out_csv_file)
        except: 
            out_df = pd.DataFrame([count_data])
    else: 
        with open(out_csv_file, "w") as f:
            f.write("class,condition,count\n")
        out_df = pd.read_csv(out_csv_file)
    
    # Load event camera data
    events = tl.load_event_data(
        event_csv_file_name) 
    
    # repnet model variables 
    strides = [1, 2, 3, 4, 8]

    
    OUT_VISUALIZATIONS_DIR = './visualization/'  
    
    # spiking variables

    width=128
    height=128
    horizontal_member_potential = np.zeros(width) 
    vertical_member_potential = np.zeros(width) 

    horizontal_neurons = np.zeros(width).reshape((1, -1))
    vertical_neurons = np.zeros(height).reshape((1, -1))

    decay = .009 # 0.009
    random_wt_2d = lambda _row, _col: np.ones((_row, _col)) * 0.5 # np.round(np.random.rand(_row, _col), 3)
    weight_input_horizontal_hidden = random_wt_2d(1, height) # np.ones((1, height)) * weight_scale_1
    weight_input_horizontal_hidden_1 = random_wt_2d(height, height//2) # np.ones((height, height//2)) * weight_scale_1

    # not considering the height and width for the image as they are symmetrical 
    weight_input_vertical_hidden = random_wt_2d(1, height) # np.ones((1, height)) * weight_scale_1
    weight_input_vertical_hidden_1 = random_wt_2d(height, height//2) # np.ones((height, height//2)) * weight_scale_1
    
    peak = 255 
    spike = 1 # 255 
    threshold = 0.8 

    fps = 120
    frame_per_second  = 1/fps 
    
    # read frames and apply preprocessing 
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),      
    ])

    raw_frames, frames = [], [] 

    for idx, frame_image in enumerate(
        e2v.get_event_frame(events, frame_per_second = frame_per_second, 
            decay= decay)):
        cv2.imshow('raw' , frame_image)
        # frame_image = frame_image.T
        
        frame_image = np.clip(frame_image * 255, 0, 255).astype(np.uint8)
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_GRAY2BGR)
        cv2.imshow('test', cv2.resize(frame_image, [frame_image.shape[0] * 10, frame_image.shape[0] * 10] )) 
        
        if(cv2.waitKey(1) == ord('q')):
            break 
        raw_frames.append(frame_image)
        frame = transform(frame_image)
        frames.append(frame)
    cv2.destroyAllWindows()
    
    
    # Load model
    model = RepNet()
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # modifying 
    best_stride, best_confidence, best_period_length, best_period_count, best_periodicity_score, best_embeddings = None, None, None, None, None, None
    for stride in strides: 
        raw_period_length, raw_periodicity_score, embeddings = [], [], []
        spikes_collection = [] 
        for idx, frame_image in enumerate(
            e2v.get_event_frame(events, frame_per_second = frame_per_second,
            decay=0.009, event_time_include=True)):
            
            if((idx % stride) == 0):
                frame_image = frame_image # .T
                horizontal_member_potential = (decay * horizontal_member_potential) +  np.dot(weight_input_horizontal_hidden, frame_image)
                # generating spikes     
                horizontal_neurons[horizontal_member_potential > (spike * threshold)] = 1 
                horizontal_neurons[horizontal_member_potential <= (spike * threshold)] = 0  
                # reset the membrane potential 
                horizontal_member_potential[horizontal_member_potential > (spike * threshold)] = 0 #        
                # prevent neurons to go to negative
                horizontal_member_potential[horizontal_member_potential < 0] = 0
                # spikes_collection.append(horizontal_neurons.copy())
                
                frame_image = frame_image.T 
                vertical_member_potential = (decay * vertical_member_potential) +  np.dot(weight_input_vertical_hidden, frame_image)
                # generating spikes     
                vertical_neurons[vertical_member_potential > (spike * threshold)] = 1 
                vertical_neurons[vertical_member_potential <= (spike * threshold)] = 0  
                # reset the membrane potential 
                vertical_member_potential[vertical_member_potential > (spike * threshold)] = 0 #        
                # prevent neurons to go to negative
                vertical_member_potential[vertical_member_potential < 0] = 0
                spikes_collection.append(np.concat([horizontal_neurons.copy(), vertical_neurons.copy()]))
    
            # process each batch separately 
            # Run inference
            with torch.no_grad():
                if((idx + 1)% 64 == 0):
                    
                    if len(spikes_collection) != 64:
                        continue
                    spikes_0 = spikes_collection
                    spikes_0 = np.array(spikes_0).reshape(-1, 128 * 2)
                    # reduced_spikes_0 = np.mean(spikes_0.reshape(-1, 64, 2), axis=2)
                    
                    x = torch.tensor(spikes_0, dtype=torch.float32, device=device).reshape(1, 64, 128 * 2)
                    batch_period_length, batch_periodicity, batch_embeddings = model(x)
                    raw_period_length.append(batch_period_length[0].cpu())
                    raw_periodicity_score.append(batch_periodicity[0].cpu())
                    embeddings.append(batch_embeddings[0].cpu())

                    spikes_collection = [] # empty the spike collection 
                ...    
        # Post-process results
        if (len(raw_period_length) and len(raw_periodicity_score)):
            raw_period_length, raw_periodicity_score, embeddings = torch.cat(raw_period_length), torch.cat(raw_periodicity_score), torch.cat(embeddings)
            confidence, period_length, period_count, periodicity_score = model.get_counts(raw_period_length, raw_periodicity_score, stride)
            if best_confidence is None or confidence > best_confidence:
                best_stride, best_confidence, best_period_length, best_period_count, best_periodicity_score, best_embeddings = stride, confidence, period_length, period_count, periodicity_score, embeddings
        ... 
    if best_stride is None:
        raise RuntimeError('The stride values used are too large and nove 64 video chunk could be sampled. Try different values for --strides.')
    print(f'Predicted a period length of {best_period_length/fps:.1f} seconds (~{int(best_period_length)} frames) with a confidence of {best_confidence:.2f} using a stride of {best_stride} frames.')

    # Generate plots and videos
    print(f'Save plots and video with counts to {OUT_VISUALIZATIONS_DIR}...')
    os.makedirs(OUT_VISUALIZATIONS_DIR, exist_ok=True)
    dist = torch.cdist(best_embeddings, best_embeddings, p=2)**2
    tsm_img = plots.plot_heatmap(dist.numpy(), log_scale=True)
    pca_img = plots.plot_pca(best_embeddings.numpy())
    cv2.imwrite(os.path.join(OUT_VISUALIZATIONS_DIR, 'tsm.png'), tsm_img)
    cv2.imwrite(os.path.join(OUT_VISUALIZATIONS_DIR, 'pca.png'), pca_img)
    
    # Generate video with counts
    rep_frames = plots.plot_repetitions(raw_frames[:len(best_period_count)], best_period_count.tolist(), best_periodicity_score.tolist() if not True else None)
    video = cv2.VideoWriter(os.path.join(OUT_VISUALIZATIONS_DIR, 'repetitions.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, rep_frames[0].shape[:2][::-1])
    for frame in rep_frames:
        video.write(frame)
    video.release()

    print('Done')
    
    count_data["count"] = round(best_period_count.tolist()[-1])
    try:
        if (out_df.loc[(out_df['class'] == event_class) & (out_df['condition'] == event_user), 'count']):
            out_df.loc[(out_df['class'] == event_class) & (out_df['condition'] == event_user), 'count'] = count_data["count"]
    except:
        out_df = pd.concat([out_df, pd.DataFrame([count_data])])
    out_df.to_csv(out_csv_file, index=False)
    
    
if __name__ == "__main__":
    # event_classes = ['class2', 'class3', 'class4', 'class5', 'class6', 'class7']
    # event_users = [
    #     "user02_fluorescent_led",
    #     "user02_fluorescent",
    #     "user02_lab",
    #     "user02_led",
    #     "user02_natural"
    # ]
    
    # for event_class in event_classes:
    #     for event_user in event_users:
    #         main(event_class, event_user)
    
    file_names = glob.glob('./event_csv/split_data/class?/*.csv')
    for file_name in file_names:
        print("[*] ---- Processing file: ", file_name)
        event_class, event_user = file_name.split('/')[-2:]
        event_user = event_user.split('.')[0] 
        main(event_class, event_user,
            out_csv_file = "artificial_encoder_replaced_output.csv")