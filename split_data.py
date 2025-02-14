import random
import torch
import numpy as np

# Set random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Shuffle the data indices for each subject
def shuffle_data(data_list):
    random.shuffle(data_list)
    return data_list

data = torch.load('/DATA/MTP_b21260/BrainVis/data/EEG/eeg_5_95_std.pth')
data1 = shuffle_data([i for i in range(len(data['dataset'])) if data['dataset'][i]['subject'] == 1])
data2 = shuffle_data([i for i in range(len(data['dataset'])) if data['dataset'][i]['subject'] == 2])
data3 = shuffle_data([i for i in range(len(data['dataset'])) if data['dataset'][i]['subject'] == 3])
data4 = shuffle_data([i for i in range(len(data['dataset'])) if data['dataset'][i]['subject'] == 4])
data5 = shuffle_data([i for i in range(len(data['dataset'])) if data['dataset'][i]['subject'] == 5])
data6 = shuffle_data([i for i in range(len(data['dataset'])) if data['dataset'][i]['subject'] == 6])

train_sp, val_sp = 0.8, 0.1  # Train, Validation split ratios

def split_data(data_list):
    train_end = int(len(data_list) * train_sp)
    val_end = train_end + int(len(data_list) * val_sp)
    
    train = data_list[:train_end]
    val = data_list[train_end:val_end]
    test = data_list[val_end:]
    
    return train, val, test

# Split the data for each subject
splits = {}
for idx, data in enumerate([data1, data2, data3, data4, data5, data6], start=1):
    train, val, test = split_data(data)
    splits[idx] = {
        'train': train,
        'val': val,
        'test': test
    }

# Concatenate all indices for each split
all_train = sum([splits[i]['train'] for i in splits], [])
all_val = sum([splits[i]['val'] for i in splits], [])
all_test = sum([splits[i]['test'] for i in splits], [])

# Save all concatenated splits at index 0
splits[0] = {
    'train': all_train,
    'val': all_val,
    'test': all_test
}

# Save to file
save_path = 'data/EEG/block_splits_by_image_all_new.pth'
torch.save({'splits': splits}, save_path)

print(f"Splits saved to {save_path}")
