import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os
from .predict_handler import predict
import pandas as pd
from .statistics import statistics_diff_box
from tqdm import tqdm


def show_predict_diff(model_file, data_folder, data_idx):
    pt_file = [file for file in os.listdir(f'result/{model_file}') if file.endswith('.pt')][0]
    model_file = f'result/{model_file}/{pt_file}'
    data_folder = f'data/generate_dataset/{data_folder}'
    # predict
    train_control, train_control_changed, train_cloud, train_cloud_changed, predict_cloud_changed, \
        diff_predict, diff_control_changed, diff_cloud_changed = predict(model_file, data_folder, data_idx)
    # visualization parameters
    train_control_changed = pv.PolyData(train_control_changed)
    train_control_changed.point_data['diff'] = diff_control_changed
    train_cloud = pv.PolyData(train_cloud)
    train_cloud = train_cloud.delaunay_3d()
    train_cloud_changed = pv.PolyData(train_cloud_changed)
    train_cloud_changed = train_cloud_changed.delaunay_3d()
    train_cloud_changed.point_data['diff'] = diff_cloud_changed
    predict_cloud_changed = pv.PolyData(predict_cloud_changed)
    predict_cloud_changed = predict_cloud_changed.delaunay_3d()
    predict_cloud_changed.point_data['diff'] = diff_predict
    point_size = 10
    opacity = 0.2
    # show one by one
    p = pv.Plotter(title='train_control')
    p.set_background('w')
    p.add_mesh(train_cloud, color='tan', opacity=opacity)
    p.add_points(train_control, color='r', render_points_as_spheres=True, point_size=point_size)
    p.show()
    p = pv.Plotter(title='train_cloud')
    p.set_background('w')
    p.add_mesh(train_cloud, color='tan')
    p.show()
    p = pv.Plotter(title='train_control_changed')
    p.set_background('w')
    p.add_mesh(train_cloud_changed, color='tan', opacity=opacity)
    p.add_points(train_control_changed, color='r', render_points_as_spheres=True, point_size=point_size,
                 scalars='diff', cmap='jet', scalar_bar_args={'color': 'black'})
    p.show()
    p = pv.Plotter(title='train_cloud_changed')
    p.set_background('w')
    p.add_mesh(train_cloud_changed, color='tan', scalars='diff', cmap='jet', scalar_bar_args={'color': 'black'})
    p.show()
    p = pv.Plotter(title='predict_cloud_changed')
    p.set_background('w')
    p.add_mesh(predict_cloud_changed, cmap='jet', scalars='diff', scalar_bar_args={'color': 'black'},
               clim=[0.00356, 0.490])
    p.show()


def show_loss_rmse(result_folders=None, legend_names=None):
    suptitles = ['train loss', 'test loss', 'train RMSE', 'test RMSE']
    plot_results = []
    module_names = []
    if result_folders is None:
        result_folders = os.listdir('result')
    n_result = len(result_folders)
    for result_folder in result_folders:
        result_folder = f'result/{result_folder}'
        plot_name = [file for file in os.listdir(result_folder) if file.startswith('plot_result')][0]
        plot_result = pd.read_csv(f'{result_folder}/{plot_name}', header=0).values
        plot_results.append(plot_result)
        if legend_names is None:
            module_name = [file for file in os.listdir(f'{result_folder}') if file.endswith('.pt')][0]
            module_names.append(module_name[:-3])
    if legend_names is not None:
        module_names = legend_names
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(suptitles[i])
        [plt.plot(plot_results[j][:, i], label=module_names[j]) for j in range(n_result)]
        plt.legend()
    plt.show()


def show_min_loss(result_folders, x_names):
    plot_results = []
    for result_folder in result_folders:
        result_folder = f'result/{result_folder}'
        plot_name = [file for file in os.listdir(result_folder) if file.startswith('plot_result')][0]
        plot_result = pd.read_csv(f'{result_folder}/{plot_name}', header=0).values[:, 0]
        plot_results.append(plot_result[-1])
    plt.figure()
    plt.plot(x_names, plot_results)
    plt.show()


def show_diff_box(data_folders, result_folders=None, legends=None, idx=None):
    if result_folders is None:
        result_folders = os.listdir('result')
    statistics_diff_box(data_folders, result_folders, idx)
    all_diff = None
    labels = []
    for r, result_folder in tqdm(enumerate(result_folders), 'load diff'):
        # load diff
        if idx is None:
            diff = pd.read_csv(f'result/{result_folder}/predict_diff.csv', header=None).values
        else:
            diff = pd.read_csv(f'result/{result_folder}/predict_diff_{idx}.csv', header=None).values
        if r == 0:
            all_diff = np.zeros([diff.shape[0], len(result_folders)])
        all_diff[:, r] = diff.flatten()
        # append label
        module_file = [file for file in os.listdir(f'result/{result_folder}') if file.endswith('.pt')][0]
        labels.append(module_file[:-3])
    if legends is not None:
        labels = legends
    plt.boxplot(all_diff, labels=labels)
    plt.show()
