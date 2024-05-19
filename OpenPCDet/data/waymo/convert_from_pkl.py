import os
import numpy as np
import pickle


def save_ply(points, path):
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(points)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for point in points:
            f.write('{} {} {}\n'.format(point[0], point[1], point[2]))

filtered_path = '/data2/hanwei/OpenPCDet/data/waymo/filtered/'
original_path = '/data2/hanwei/OpenPCDet/data/waymo/waymo_processed_data_v0_5_0_original'
save_path = '/data2/hanwei/OpenPCDet/data/waymo/waymo_processed_data_v0_5_0_oracle'

folder_list = os.listdir(filtered_path)

for folder_name in folder_list:
    if folder_name[0]!='s':
        continue
    folder_path = os.path.join(filtered_path, folder_name)
    print(folder_name)
    file_list = os.listdir(folder_path)
    #print(file_list)
    for file_name in file_list:
        if file_name.endswith('.pkl'):
            file_path = os.path.join(folder_path, file_name)
            file_idx = file_name.split('_')[1].split('.')[0]
            save_name = '{:04d}'.format(int(file_idx)-1) + '.npy'
            save_npy_path = os.path.join(os.path.join(save_path, folder_name), save_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                data = data[:,[3,4,5,1,2,0]]
                data[:,5] = -1
            np.save(save_npy_path, data)
