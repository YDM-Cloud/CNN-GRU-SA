import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from modules.cnn_gru import CNN_GRU
from modules.cnn_gru_cbam import CNN_GRU_CBAM
from modules.cnn_gru_sa import CNN_GRU_SA
from modules.cnn_lstm_sa import CNN_LSTM_SA


def statistics_diff_box(data_folders, result_folders, idx=None):
    # calculate diff
    for r, (data_folder, result_folder) in enumerate(zip(data_folders, result_folders)):
        # check box result
        if idx is None:
            diff_file = 'predict_diff.csv'
        else:
            diff_file = f'predict_diff_{idx}.csv'
        result_folder = f'result/{result_folder}'
        if diff_file in os.listdir(result_folder):
            continue
        # load dataset
        data_folder = f'data/generate_dataset/{data_folder}/'
        control = torch.load(data_folder + 'control.pt')
        control_changed = torch.load(data_folder + 'control_changed.pt')
        cloud = torch.load(data_folder + 'cloud.pt')
        cloud_changed = torch.load(data_folder + 'cloud_changed.pt')
        mean = torch.load(data_folder + 'mean.pt').numpy()
        std = torch.load(data_folder + 'std.pt').numpy()
        n_train = control.shape[0]
        diff = np.zeros([n_train, cloud.shape[1]])
        # load model
        Model = [file for file in os.listdir(result_folder) if file.endswith('.pt')][0]
        model = eval(f'{Model[:-3]}(device="cpu")')
        model.load_state_dict((torch.load(f'{result_folder}/{Model}', map_location='cpu')))
        model.eval()
        # predict
        if idx is not None:
            with torch.no_grad():
                para1 = control[idx].unsqueeze(0)
                para2 = control_changed[idx].unsqueeze(0)
                para3 = cloud[idx].unsqueeze(0)
                predict_output = model(para1, para2, para3)
            # inverse standardization
            predict_cloud_changed = (predict_output.numpy() * std + mean)[0]
            train_cloud_changed = cloud_changed[idx].numpy() * std + mean
            diff = np.linalg.norm(predict_cloud_changed - train_cloud_changed, axis=1)
            print(f'progress: {r + 1}/{len(result_folders)}')
        else:
            for i in tqdm(range(n_train), f'{r + 1}/{len(result_folders)}'):
                with torch.no_grad():
                    para1 = control[i].unsqueeze(0)
                    para2 = control_changed[i].unsqueeze(0)
                    para3 = cloud[i].unsqueeze(0)
                    predict_output = model(para1, para2, para3)
                # inverse standardization
                predict_cloud_changed = (predict_output.numpy() * std + mean)[0]
                train_cloud_changed = cloud_changed[i].numpy() * std + mean
                diff[i, :] = np.linalg.norm(predict_cloud_changed - train_cloud_changed, axis=1)
        # save csv
        diff = pd.DataFrame(diff.flatten())
        diff.to_csv(f'{result_folder}/{diff_file}', header=False, index=False)


def statistics_outlier_count(result_folders, threshold=0.1):
    for i, result_folder in enumerate(result_folders):
        data = pd.read_csv(f'result/{result_folder}/predict_diff.csv', header=None).values.flatten()
        n_data = data.shape[0]
        n_outlier = data[data >= threshold].shape[0]
        print(f'[{i + 1}/{len(result_folders)}] n_data: {n_data}, n_outlier: {n_outlier}')


def statistics_outlier_mean(result_folders, threshold=0.1):
    for i, result_folder in enumerate(result_folders):
        data = pd.read_csv(f'result/{result_folder}/predict_diff.csv', header=None).values.flatten()
        mean = np.mean(data[data >= threshold])
        print(f'[{i + 1}/{len(result_folders)}] mean: {mean}')


def statistics_run_time(data_folder, result_folder):
    # check box result
    diff_file = 'statistics_run_time.csv'
    result_folder = f'result/{result_folder}'
    # load dataset
    data_folder = f'data/generate_dataset/{data_folder}/'
    control = torch.load(data_folder + 'control.pt')
    control_changed = torch.load(data_folder + 'control_changed.pt')
    cloud = torch.load(data_folder + 'cloud.pt')
    n_train = control.shape[0]
    diff = np.zeros(n_train)
    # load model
    Model = [file for file in os.listdir(result_folder) if file.endswith('.pt')][0]
    model = eval(f'{Model[:-3]}(device="cpu")')
    model.load_state_dict((torch.load(f'{result_folder}/{Model}', map_location='cpu')))
    model.eval()
    # predict
    for i in tqdm(range(n_train)):
        start_stamp = time.time()
        with torch.no_grad():
            para1 = control[i].unsqueeze(0)
            para2 = control_changed[i].unsqueeze(0)
            para3 = cloud[i].unsqueeze(0)
            predict_output = model(para1, para2, para3)
        diff[i] = time.time() - start_stamp
    # save csv
    diff = pd.DataFrame(diff)
    diff.to_csv(diff_file, header=False, index=False)
