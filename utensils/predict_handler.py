import torch
import numpy as np
import os
from modules.cnn_gru import CNN_GRU
from modules.cnn_gru_cbam import CNN_GRU_CBAM
from modules.cnn_gru_sa import CNN_GRU_SA
from modules.cnn_lstm_sa import CNN_LSTM_SA


def predict(model_file, data_folder, data_idx):
    # load model
    Model = os.path.basename(model_file)[:-3]
    model = eval(f'{Model}(device="cpu")')
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
    # load dataset
    data_folder += '/'
    control = torch.load(data_folder + 'control.pt')[data_idx]
    control_changed = torch.load(data_folder + 'control_changed.pt')[data_idx]
    cloud = torch.load(data_folder + 'cloud.pt')[data_idx]
    cloud_changed = torch.load(data_folder + 'cloud_changed.pt')[data_idx]
    mean = torch.load(data_folder + 'mean.pt').numpy()
    std = torch.load(data_folder + 'std.pt').numpy()
    # predict
    with torch.no_grad():
        para1 = control.unsqueeze(0)
        para2 = control_changed.unsqueeze(0)
        para3 = cloud.unsqueeze(0)
        predict_output = model(para1, para2, para3)
    # inverse standardization
    predict_cloud_changed = (predict_output.numpy() * std + mean)[0]
    train_cloud_changed = cloud_changed.numpy() * std + mean
    train_cloud = cloud.numpy() * std + mean
    train_control = control.numpy() * std + mean
    train_control_changed = control_changed.numpy() * std + mean
    diff_predict = np.linalg.norm(predict_cloud_changed - train_cloud_changed, axis=1)
    diff_cloud_changed = np.linalg.norm(train_cloud - train_cloud_changed, axis=1)
    diff_control_changed = np.linalg.norm(train_control - train_control_changed, axis=1)
    return train_control, train_control_changed, train_cloud, train_cloud_changed, predict_cloud_changed, \
        diff_predict, diff_control_changed, diff_cloud_changed
