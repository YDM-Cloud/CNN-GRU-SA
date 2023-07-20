from utensils.train_handler import run
from modules.cnn_gru import CNN_GRU
from modules.cnn_gru_cbam import CNN_GRU_CBAM
from modules.cnn_gru_sa import CNN_GRU_SA
from modules.cnn_lstm_sa import CNN_LSTM_SA

if __name__ == '__main__':
    data_folder = 'n_train(100000)n_control(50)n_cloud(1000)point_only(1)n_loss_control(0,0.5)standardization(1)'
    run(CNN_GRU, data_folder)
    run(CNN_GRU_CBAM, data_folder)
    run(CNN_GRU_SA, data_folder)
    run(CNN_LSTM_SA, data_folder)
