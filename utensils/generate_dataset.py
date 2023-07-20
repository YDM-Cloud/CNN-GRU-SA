import os
import numpy as np
import torch
import pandas as pd
from utensils.others import get_indices
from tqdm import tqdm


def get_dataset(file, offset=True):
    with open(file) as f:
        lines = f.readlines()
        n_line = len(lines)
        origin_idx = [i for i in range(n_line) if lines[i].find('NODE OUTPUT') >= 0]
        origin_start_idx = min(origin_idx) + 1
        n_indices = abs(origin_idx[0] - origin_idx[1]) - 1
        offset_idx = [i for i in range(n_line) if lines[i].find('NODAL DISPLACEMENTS') >= 0]
        offset_start_idx = min(offset_idx) + 6
        origin_dataset = np.zeros([n_indices, 3])
        offset_dataset = np.zeros_like(origin_dataset)
        for i in range(n_indices):
            origin_dataset[i] = [float(s) for s in lines[i + origin_start_idx][:-1].split(' ') if s != ''][1:4]
            offset_dataset[i] = [float(s) for s in lines[i + offset_start_idx][:-1].split(' ') if s != ''][1:4]
        if offset:
            return np.round(origin_dataset + offset_dataset, 2)
        else:
            return np.round(origin_dataset, 2)


def generate_dataset(n_train=None, base_dir=r'data/origin_ansys', output_folder=r'data/generate_dataset/',
                     control_dir=r'data/origin_geometry/control positions.txt'):
    # init files
    print('init files...')
    files = os.listdir(base_dir)
    control_positions = np.round(pd.read_csv(control_dir, header=None).values, 2)
    origin_cloud = get_dataset(f'{base_dir}/{files[0]}', False)
    control_indices = get_indices(origin_cloud, control_positions)
    # init dataset
    print('init dataset...')
    if n_train is None:
        n_train = len(files) - 1
    n_control = control_positions.shape[0]
    n_cloud = origin_cloud.shape[0]
    control = torch.tile(torch.Tensor(control_positions), [n_train, 1, 1])
    control_changed = torch.zeros([n_train, n_control, 3])
    cloud = torch.tile(torch.Tensor(origin_cloud), [n_train, 1, 1])
    cloud_changed = torch.zeros([n_train, n_cloud, 3])
    # fill data to dataset
    for i in tqdm(range(n_train), 'fill data'):
        temp_cloud_changed = get_dataset(f'{base_dir}/{files[i + 1]}')
        temp_cloud_changed = torch.Tensor(temp_cloud_changed)
        control_changed[i, :, :] = temp_cloud_changed[control_indices, :]
        cloud_changed[i, :, :] = temp_cloud_changed
    # standardization
    mean = torch.zeros([n_train, 4])
    std = torch.zeros([n_train, 4])
    for i in tqdm(range(n_train), 'standardization'):
        std[i, 0], mean[i, 0] = torch.std_mean(control[i])
        std[i, 1], mean[i, 1] = torch.std_mean(control_changed[i])
        std[i, 2], mean[i, 2] = torch.std_mean(cloud[i])
        std[i, -1], mean[i, -1] = torch.std_mean(cloud_changed[i])
    mean = torch.mean(mean)
    std = torch.mean(std)
    control = (control - mean) / std
    control_changed = (control_changed - mean) / std
    cloud = (cloud - mean) / std
    cloud_changed = (cloud_changed - mean) / std
    # save dataset to .pt
    print('save dataset to .pt...')
    point_only = True
    n_loss_control = 0
    loss_control_percent = 0.5
    standardization = True
    folder_name = f'n_train({n_train})n_control({n_control})n_cloud({n_cloud})point_only({str(int(point_only))})n_loss_control({n_loss_control},{round(loss_control_percent, 1)})standardization({str(int(standardization))})'
    output_folder += folder_name + '/'
    os.makedirs(output_folder)
    torch.save(control, output_folder + 'control.pt')
    torch.save(control_changed, output_folder + 'control_changed.pt')
    torch.save(cloud, output_folder + 'cloud.pt')
    torch.save(cloud_changed, output_folder + 'cloud_changed.pt')
    torch.save(mean, output_folder + 'mean.pt')
    torch.save(std, output_folder + 'std.pt')
    print('complete!')
