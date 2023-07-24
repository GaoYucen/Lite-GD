#%%
import numpy as np
import argparse
import networkx as nx
from tqdm import tqdm
import itertools
from haversine import haversine

#%% function definition
# 读取travel信息
def travel_read_ds(file_name):
    travel_dict = {}
    repeat_travel_list = []
    with open(file_name, 'r') as file_0:
        for line in tqdm(file_0):
            [travel_id, geo, link_id] = line.split()
            geo_list = geo.split(';')
            link_list_initial = link_id.split(';')
            link_list=[]
            for link in link_list_initial:
                # link_list.append(link.split(',')) #得到string list
                link_list.append(list(map(int, link.split(',')))) #利用map转化list元素为int
            travel_id = int(travel_id)
            if(travel_id in travel_dict.keys()):
                repeat_travel_list.append(travel_id)
            travel_dict[travel_id] = [geo_list, link_list]

    return travel_dict, repeat_travel_list

# 读取label信息
def label_read_ds(file_name):
    label_dict = {}
    repeat_label_list = []
    # three_carpool_list = []
    with open(file_name, 'r') as file_0:
        for line in tqdm(file_0):
            if len(line.split()) == 5:
                [travel_id, waypoint_list, driver_loc, order_loc0, order_loc1] = line.split()
            elif len(line.split()) == 6:
                [travel_id, waypoint_list, driver_loc, order_loc0, order_loc1, order_loc2] = line.split()
            travel_id = int(travel_id)
            if travel_id in label_dict.keys():
                repeat_label_list.append(travel_id)
            if len(line.split()) == 5:
                waypoint_list = [int(x) for x in waypoint_list.split(',')]
                geo_list = []
                geo_list.append(driver_loc[1:-1])
                order_loc0 = order_loc0.split('to')
                order_loc1 = order_loc1.split('to')
                for x in order_loc0:
                    geo_list.append(x[1:-1])
                for x in order_loc1:
                    geo_list.append(x[1:-1])
                label_dict[travel_id] = [waypoint_list, geo_list]
            elif len(line.split()) == 6:
                waypoint_list = [int(x) for x in waypoint_list.split(',')]
                geo_list = []
                geo_list.append(driver_loc[1:-1])
                order_loc0 = order_loc0.split('to')
                order_loc1 = order_loc1.split('to')
                order_loc2 = order_loc2.split('to')
                for x in order_loc0:
                    geo_list.append(x[1:-1])
                for x in order_loc1:
                    geo_list.append(x[1:-1])
                for x in order_loc2:
                    geo_list.append(x[1:-1])
                label_dict[travel_id] = [waypoint_list, geo_list]

    return label_dict, repeat_label_list

# dict_verify 输出非一致的list，并删除非一致订单
def dict_verify(travel_dict, label_dict):
    no_consistent_list = []
    no_consistent_type = []
    travel_in_line = []
    for travel_id in travel_dict.keys():
        depart_list = [-1, 0, 2, 4]
        label_depart_list = [0, 1, 3, 5]
        verify_list = [0, 0, 0, 0]
        if len(label_dict[travel_id][0]) != len(travel_dict[travel_id][0]):
            no_consistent_list.append(travel_id)
            no_consistent_type.append(1)
        elif len(label_dict[travel_id][0]) == 5:
            for x in range(0, 3):
                verify_list[x] = travel_dict[travel_id][0][label_dict[travel_id][0].index(depart_list[x])] == \
                                 label_dict[travel_id][1][label_depart_list[x]]
            if not(verify_list[0] & verify_list[1] & verify_list[2]):
                no_consistent_list.append(travel_id)
                no_consistent_type.append(2)
        elif len(label_dict[travel_id][0]) == 7:
            for x in range(0, 4):
                verify_list[x] = travel_dict[travel_id][0][label_dict[travel_id][0].index(depart_list[x])] == \
                                 label_dict[travel_id][1][label_depart_list[x]]
            if not(verify_list[0] & verify_list[1] & verify_list[2] & verify_list[3]):
                no_consistent_list.append(travel_id)
                no_consistent_type.append(2)

    for travel_id in no_consistent_list:
        travel_dict.pop(travel_id)
        label_dict.pop(travel_id)

    return travel_dict, label_dict

# 调整travel_dict的顺序到-1,0,2,4,1,3,5
def adjust_order(travel_dict, label_dict):
    for travel_id in travel_dict.keys():
        geo_list = []
        can_link_list = []
        if len(label_dict[travel_id][0]) == 5:
            waypoint_list = [-1, 0, 2, 1, 3]
            for x in waypoint_list:
                geo_list.append(travel_dict[travel_id][0][label_dict[travel_id][0].index(x)])
                can_link_list.append(travel_dict[travel_id][1][label_dict[travel_id][0].index(x)])
        elif len(label_dict[travel_id][0]) == 7:
            waypoint_list = [-1, 0, 2, 4, 1, 3, 5]
            for x in waypoint_list:
                geo_list.append(travel_dict[travel_id][0][label_dict[travel_id][0].index(x)])
                can_link_list.append(travel_dict[travel_id][1][label_dict[travel_id][0].index(x)])
        travel_dict[travel_id][0] = geo_list
        travel_dict[travel_id][1] = can_link_list

    return travel_dict

# 删除行中拼
def del_in_line(travel_dict, label_dict):
    for travel_id in list(travel_dict.keys()):
        if label_dict[travel_id][0][0] != -1:
            travel_dict.pop(travel_id)
            label_dict.pop(travel_id)

    return travel_dict, label_dict

#%%
def data_con(mat_len, T, PT, start, length, travel_dict):
    # 矩阵声明
    nodes_num_mat = []
    label_mat = []
    label_len_mat = []
    V_mat = []
    E_sd_mat = []
    E_ed_mat = []
    V_pt_mat = []
    V_dt_mat = []
    V_ft_mat = []
    V_num_mat = []
    V_dispatch_mask_mat = []
    E_mask_mat = []
    start_idx_mat = []
    E_pt_dif_mat = []
    E_df_dif_mat = []
    cou_mat = []
    V_reach_mask_mat = []
    A_mat = []

    for i in range(start, start+length):
        travel_id = list(travel_dict.keys())[i]
        waypoint_list = []
        nodes_num = 0
        if len(label_dict[travel_id][0]) == 5:
            waypoint_list = [0] * 5
            waypoint_list[label_dict[travel_id][0].index(0)] = 1
            waypoint_list[label_dict[travel_id][0].index(2)] = 2
            waypoint_list[label_dict[travel_id][0].index(1)] = 3
            waypoint_list[label_dict[travel_id][0].index(3)] = 4
            nodes_num = 5
        elif len(label_dict[travel_id][0]) == 7:
            waypoint_list = [0] * 7
            waypoint_list[label_dict[travel_id][0].index(0)] = 1
            waypoint_list[label_dict[travel_id][0].index(2)] = 2
            waypoint_list[label_dict[travel_id][0].index(1)] = 3
            waypoint_list[label_dict[travel_id][0].index(3)] = 4
            waypoint_list[label_dict[travel_id][0].index(4)] = 5
            waypoint_list[label_dict[travel_id][0].index(5)] = 6
            nodes_num = 7
        nodes_num_mat.append(nodes_num)

        label = waypoint_list[1:]
        for i in range(mat_len - len(label)):
            label.append(8)
        label_mat.append([label, [8] * mat_len, [8] * mat_len])

        if len(label_dict[travel_id][0]) == 5:
            label_len = [4, 0, 0]
        else:
            label_len = [6, 0, 0]
        label_len_mat.append(label_len)

        V = []
        for i in range(0, len(waypoint_list)):
            V.append(
                [float(travel_dict[travel_id][0][i].split(',')[0]), float(travel_dict[travel_id][0][i].split(',')[1]),
                 PT])
        for i in range(0, mat_len - len(waypoint_list)):
            V.append([0, 0, 0])
        V_mat.append(V)

        E_sd = [[0] * mat_len] * mat_len

        for i in range(nodes_num):
            for j in range(nodes_num):
                E_sd[i][j] = haversine(V[i][0:2][::-1], V[j][0:2][::-1]) * 1000
                E_sd[j][i] = haversine(V[i][0:2][::-1], V[j][0:2][::-1]) * 1000
        E_sd_mat.append(E_sd)
        E_ed_mat.append(E_sd)

        V_pt = [PT] * mat_len
        V_pt_mat.append(V_pt)

        V_dt = [0] * mat_len
        V_dt_mat.append(V_dt)

        V_ft = [0] * mat_len
        time = 0
        for i in range(nodes_num - 1):
            time += E_sd[waypoint_list[i]][waypoint_list[i + 1]]
            V_ft[waypoint_list[i + 1]] = time
        V_ft_mat.append(V_ft)

        V_num = [[0] * mat_len]
        for i in range(1, nodes_num):
            V_num[0][i] = 1
        for i in range(T - 1):
            V_num.append([0] * mat_len)
        V_num_mat.append(V_num)

        V_dispatch_mask = [[0] * mat_len]
        for i in range(1, nodes_num):
            V_dispatch_mask[0][i] = 1
        for i in range(T - 1):
            V_dispatch_mask.append([0] * mat_len)
        V_dispatch_mask_mat.append(V_dispatch_mask)

        E_mask = np.zeros([T, mat_len, mat_len])
        for i in range(1, nodes_num):
            for j in range(1, nodes_num):
                E_mask[0, i, j] = 1
        E_mask_mat.append(E_mask)

        start_idx = [0] * T
        start_idx_mat.append(start_idx)

        E_pt_dif = np.zeros([mat_len, mat_len])
        E_pt_dif_mat.append(E_pt_dif)

        E_dt_dif = np.zeros([mat_len, mat_len])
        E_df_dif_mat.append(E_dt_dif)

        cou = [1, 1, 1, 3]
        cou_mat.append(cou)

        V_reach_mask = np.array([True] * (T * mat_len)).reshape(T, mat_len)
        for i in range(nodes_num):
            if i % 2 != 0:
                V_reach_mask[0, i] = False
        V_reach_mask_mat.append(V_reach_mask)

        A = np.zeros([T, mat_len, mat_len])
        for i in range(nodes_num):
            for j in range(nodes_num):
                if (i == 0) & (j == 0):
                    A[0, i, j] = -1
                if (V_reach_mask[0, i] == False) & (i == j):
                    A[0, i, j] = -1
                if (i == 0) & (V_reach_mask[0, j] == False):
                    A[0, i, j] = 1
                if (V_reach_mask[0, i] == False) & (j == 0):
                    A[0, i, j] = 1
        A_mat.append(A)

    # 构造数据
    data_new = {'nodes_num': np.array(nodes_num_mat)}
    data_new['label'] = np.array(label_mat)
    data_new['label_len'] = np.array(label_len_mat)
    data_new['V'] = np.array(V_mat)
    data_new['E_ed'] = np.array(E_ed_mat)
    data_new['E_sd'] = np.array(E_sd_mat)
    data_new['V_pt'] = np.array(V_pt_mat)
    data_new['V_dt'] = np.array(V_dt_mat)
    data_new['V_ft'] = np.array(V_ft_mat)
    data_new['V_num'] = np.array(V_num_mat)
    data_new['V_dispatch_mask'] = np.array(V_dispatch_mask_mat)
    data_new['E_mask'] = np.array(E_mask_mat)
    data_new['start_idx'] = np.array(start_idx_mat)
    data_new['pt_dif'] = np.array(E_pt_dif_mat)
    data_new['dt_dif'] = np.array(E_df_dif_mat)
    data_new['cou'] = np.array(cou_mat)
    data_new['V_reach_mask'] = np.array(V_reach_mask_mat)
    data_new['A'] = np.array(A_mat)

    return data_new

def data_save(filepath, param, s_idx, e_idx, all_seg, travel_dict):
    nodes_num_mat = np.empty(int(len(travel_dict) / all_seg) * (e_idx - s_idx))
    label_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), T, mat_len))
    label_len_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), T))
    V_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), mat_len, T))
    E_ed_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), mat_len, mat_len))
    E_sd_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), mat_len, mat_len))
    V_pt_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), mat_len))
    V_dt_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), mat_len))
    V_ft_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), mat_len))
    V_num_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), T, mat_len))
    V_dispatch_mask_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), T, mat_len))
    E_mask_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), T, mat_len, mat_len))
    start_idx_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), T))
    pt_dif_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), mat_len, mat_len))
    dt_dif_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), mat_len, mat_len))
    cou_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), 4))
    V_reach_mask_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), T, mat_len))
    A_mat = np.empty((int(len(travel_dict) / all_seg) * (e_idx - s_idx), T, mat_len, mat_len))

    for i in range(int((len(travel_dict) / 20)) * 20):
        if (i % all_seg >= s_idx) & (i % all_seg < e_idx):
            nodes_num_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['nodes_num'][i]
            label_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['label'][i]
            label_len_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['label_len'][i]
            V_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['V'][i]
            E_ed_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['E_ed'][i]
            E_sd_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['E_sd'][i]
            V_pt_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['V_pt'][i]
            V_dt_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['V_dt'][i]
            V_ft_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['V_ft'][i]
            V_num_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['V_num'][i]
            V_dispatch_mask_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['V_dispatch_mask'][i]
            E_mask_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['E_mask'][i]
            start_idx_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['start_idx'][i]
            pt_dif_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['pt_dif'][i]
            dt_dif_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['dt_dif'][i]
            cou_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['cou'][i]
            V_reach_mask_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['V_reach_mask'][i]
            A_mat[int(i / all_seg) * (e_idx - s_idx) + i % all_seg - s_idx] = data['A'][i]

    # 构造数据
    data_new = {'nodes_num': np.array(nodes_num_mat)}
    data_new['label'] = np.array(label_mat)
    data_new['label_len'] = np.array(label_len_mat)
    data_new['V'] = np.array(V_mat)
    data_new['E_ed'] = np.array(E_ed_mat)
    data_new['E_sd'] = np.array(E_sd_mat)
    data_new['V_pt'] = np.array(V_pt_mat)
    data_new['V_dt'] = np.array(V_dt_mat)
    data_new['V_ft'] = np.array(V_ft_mat)
    data_new['V_num'] = np.array(V_num_mat)
    data_new['V_dispatch_mask'] = np.array(V_dispatch_mask_mat)
    data_new['E_mask'] = np.array(E_mask_mat)
    data_new['start_idx'] = np.array(start_idx_mat)
    data_new['pt_dif'] = np.array(pt_dif_mat)
    data_new['dt_dif'] = np.array(dt_dif_mat)
    data_new['cou'] = np.array(cou_mat)
    data_new['V_reach_mask'] = np.array(V_reach_mask_mat)
    data_new['A'] = np.array(A_mat)

    np.save(filepath + param + '.npy', data_new)

#%% read travel and label
city_list = ['shanghai_big', 'qingdao', 'shanghai']
city = city_list[2]

filepath = '/Users/gaoyucen/滴滴项目（数据）/'

if city == 'shanghai':
    # read travel
    travel_dict = travel_read_ds(filepath + 'carpool_route_point_shanghai_20220808_res_add_multi_link_valid_ds')[0]

    # read label
    label_dict = label_read_ds(filepath + 'carpool_route_point_shanghai_20220808_ds')[0]

    # dict verify
    travel_dict, label_dict = dict_verify(travel_dict, label_dict)

    # 调整travel_dict顺序
    travel_dict = adjust_order(travel_dict, label_dict)

    # 删除行中拼
    travel_dict, label_dict = del_in_line(travel_dict, label_dict)

elif city == 'shanghai_big':
    # read travel
    travel_dict = travel_read_ds(filepath + 'shanghaidata/carpool_route_point_shanghai_20220915_0930_res_add_multi_link_valid_ds')[0]

    # read label
    label_dict = label_read_ds(filepath + 'shanghaidata/carpool_route_point_shanghai_20220915_0930_ds')[0]

    # dict verify
    travel_dict, label_dict = dict_verify(travel_dict, label_dict)

    # 调整travel_dict顺序
    travel_dict = adjust_order(travel_dict, label_dict)

    # 删除行中拼
    travel_dict, label_dict = del_in_line(travel_dict, label_dict)

elif city == 'qingdao':
    # read travel
    travel_dict = travel_read_ds(filepath + 'carpool_route_point_qingdao_20221010_res_add_multi_link_valid_ds')[0]

    # read label
    label_dict = label_read_ds(filepath + 'carpool_route_point_qingdao_20221010_ds')[0]

    # dict verify
    travel_dict, label_dict = dict_verify(travel_dict, label_dict)

    # 调整travel_dict顺序
    travel_dict = adjust_order(travel_dict, label_dict)

    # 删除行中拼
    travel_dict, label_dict = del_in_line(travel_dict, label_dict)

#%%
print(len(list(travel_dict.keys())))

#%% 超参数
max_task_num = 7
mat_len = max_task_num + 2
T = int((mat_len-2)/2)
PT = 1000000
start = 0
length = len(travel_dict)

#%%
data = data_con(mat_len, T, PT, start, length, travel_dict)

#%%
all_seg = 20

savefilepath = 'data/dataset/food_pd/'
param = 'train'
s_idx = 0
e_idx = 18

data_save(savefilepath, param, s_idx, e_idx, all_seg, travel_dict)

savefilepath = 'data/dataset/food_pd/'
param = 'val'
s_idx = 18
e_idx = 19

data_save(savefilepath, param, s_idx, e_idx, all_seg, travel_dict)

savefilepath = 'data/dataset/food_pd/'
param = 'test'
s_idx = 19
e_idx = 20

data_save(savefilepath, param, s_idx, e_idx, all_seg, travel_dict)


#%%
# #%%
# train_dict = {}
# val_dict = {}
# test_dict = {}
#
# for i, travel_id in enumerate(list(travel_dict.keys())):
#     if i % 18 < 16:
#         train_dict.update({travel_id: travel_dict[travel_id]})
#     elif i % 18 == 16:
#         val_dict.update({travel_id: travel_dict[travel_id]})
#     elif i % 18 == 17:
#         test_dict.update({travel_id: travel_dict[travel_id]})
#
#
# #%% 超参数
# max_task_num = 7
# mat_len = max_task_num + 2
# T = int((mat_len-2)/2)
# PT = 1000000
#
# #%% train
# savefilepath = 'data/dataset/food_pd/'
# param = 'train'
# start = 0
# length = len(train_dict)
#
# data_con(mat_len, T, PT, start, length, savefilepath, param, train_dict)
#
# # valid
# savefilepath = 'data/dataset/food_pd/'
# param = 'val'
# start = 0
# length = len(val_dict)
#
# data_con(mat_len, T, PT, start, length, savefilepath, param, val_dict)
#
# # test
# savefilepath = 'data/dataset/food_pd/'
# param = 'test'
# start = 0
# length = len(test_dict)
#
# data_con(mat_len, T, PT, start, length, savefilepath, param, test_dict)

# data_new = np.load(savefilepath+'train.npy', allow_pickle=True).item()


