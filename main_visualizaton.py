from utensils.visualization_handler import show_loss_rmse

if __name__ == '__main__':
    show_loss_rmse([
        '2023-06-20_22-32-59-876223_n_train(10000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)',
        '2023-06-21_14-27-30-068874_n_train(20000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)',
        '2023-06-22_18-50-40-781534_n_train(30000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)',
        '2023-06-24_21-14-43-482844_n_train(40000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)',
        '2023-06-26_20-59-23-755425_n_train(50000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)',
        '2023-06-29_12-24-59-936594_n_train(60000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)',
        '2023-07-01_16-45-30-596138_n_train(70000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)',
        '2023-07-05_04-08-51-765230_n_train(80000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)',
        '2023-07-05_10-30-56-374880_n_train(90000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)',
        '2023-07-08_15-46-47-352474_n_train(100000)n_control(20)n_cloud(1208)point_only(1)n_loss_control(0,0.5)standardization(1)'
    ], [i * 1e4 for i in range(1, 11)])
