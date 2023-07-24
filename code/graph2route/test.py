import os
from pprint import *
from haversine import haversine
import math
import torch

def get_workspace():
    """
    get the workspace path, i.e., the root directory of the project
    """
    cur_path = os.path.abspath(__file__)
    print('1')
    print(cur_path)
    file = os.path.dirname(cur_path)
    print(file)
    file = os.path.dirname(file)
    print(file)
    return file

def get_dataset_path(params = {}):
    """
    get file path of train, validate and test dataset
    """
    if params['model'] == 'graph2route_pd':
        dataset = 'food_pd'
    else:
        dataset = 'logistics'
    params['dataset'] = dataset
    file = ws + f'/data/dataset/{dataset}'
    train_path = file + f'/train.npy'
    val_path = file + f'/val.npy'
    test_path = file + f'/test.npy'
    return train_path, val_path, test_path

#%%
ws = get_workspace()

#%%
params={}
params['model'] = 'graph2route_pd'
print(get_dataset_path(params))

#%%
mode='train'
path_key = {'train': 'train_path', 'val': 'val_path', 'test': 'test_path'}[mode]
print(path_key)

#%%
import numpy as np
data = np.load('data/dataset/food_pd/train.npy', allow_pickle=True).item()

#%%
index = 0
nodes_num = data['nodes_num'][index]
steps = int((nodes_num-1)/2)

#%%
print(data['start_idx'][index][0:steps])
print(data['label'][index][0:steps,0:nodes_num])
print(data['label_len'][index][0:steps])

#%%
print(data['V_dispatch_mask'][index][0:steps,0:nodes_num])
print(data['E_mask'][index][0:steps,0:nodes_num, 0:nodes_num])


#%%
print(data['V_num'][index][0:steps,0:nodes_num])

#%%
# print(data['V_reach_mask'][index][0:steps,0:nodes_num])
print(data['start_idx'][index][0:steps])
for i in range(0, steps):
    for j in range(0, nodes_num):
        if data['V_reach_mask'][index][i,j] == False:
            print(0, end = ' ')
        else:
            print(1, end = ' ')
    print(end = '\n')

#%%
print(data['A'][index][0:steps,0:nodes_num, 0:nodes_num])


#%%
print(data['V_pt'][index][0:nodes_num])
print(data['V_ft'][index][0:nodes_num])
print(data['V_dt'][index][0:nodes_num])
#%%
print(data['pt_dif'][index][0:nodes_num, 0:nodes_num])
print(data['dt_dif'][index][0:nodes_num, 0:nodes_num])

#%%













# #%%
# V_reach_mask = data['V_reach_mask']
#
# B = V_reach_mask.shape[0]
# T = 12
# N = 27
# batch_V_reach_mask = V_reach_mask.reshape(B * T, N)
#
# #%%
# V=torch.tensor(data['V'])
# print(V)
# #%%
# V_1=V.permute(1, 0, 2)


# #%% 和dis矩阵对比
# dis_mat = np.zeros([nodes_num, nodes_num])
# for i in range(0, nodes_num):
#     for j in range(0, nodes_num):
#         dis_mat[i,j] = haversine([data['V'][index][i][1],data['V'][index][i][0]],[data['V'][index][j][1],data['V'][index][j][0]])
# #%%
# print(dis_mat,file = f)
# print(data['E_ed'][index][0:nodes_num,0:nodes_num], file = f)
# print(data['E_sd'][index][0:nodes_num,0:nodes_num], file = f)
# f.close()
#
# #%% man_dis_mat
# man_dis_mat = np.zeros([nodes_num, nodes_num])
# for i in range(0, nodes_num):
#     for j in range(0, nodes_num):
#         man_dis_mat[i,j] = math.sqrt(math.pow(haversine([data['V'][index][i][1],data['V'][index][i][0]],[data['V'][index][i][1],data['V'][index][j][0]]),2) + math.pow(haversine([data['V'][index][i][1],data['V'][index][j][0]],[data['V'][index][j][1],data['V'][index][j][0]]),2))
# #%%
# filename = 'data.txt'
# f = open(filename, 'w')
# print(man_dis_mat,file = f)
# print(data['E_ed'][index][0:nodes_num,0:nodes_num], file = f)
# f.close()
# print(data['E_mask'][0][6][0:nodes_num, 0:nodes_num])
# print(data['E_ed'][index][0:nodes_num,0:nodes_num])
# print(data['E_sd'][index][0:nodes_num,0:nodes_num])