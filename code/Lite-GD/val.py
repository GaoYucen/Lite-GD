#%%
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import math

# 添加路径
import sys
sys.path.append('code/Lite-GD/')
from PointerNet import PointerNet
from Data_Generator import TSPDataset
from config import get_config
import time

#%% 读取参数
params, _ = get_config()

#%% 根据是否使用gpu定义device
if params.gpu and params.sys == 'win' and torch.cuda.is_available():
    USE_CUDA = True
    USE_MPS = False
    device = torch.device('cuda:0')
    print('Using GPU, %i devices.' % torch.cuda.device_count())
elif params.gpu and params.sys == 'mac':
    USE_CUDA = False
    USE_MPS = True
    device = torch.device('mps')
    print('Using MPS')
else:
    USE_CUDA = False
    USE_MPS = False
    device = torch.device('cpu')
    print('Using CPU')

#%% 确定model参数
model = PointerNet(params.embedding_size,
                   params.hiddens,
                   params.nof_lstms,
                   params.dropout,
                   params.bidir)

# #%%
# params.test_flag = True
#
# #%% test mode
# if params.test_flag == True:

#%%
print('Test Mode!')
model.load_state_dict(torch.load('code/Lite-GD/param/param_'+str(params.nof_points)+'_'+str(params.nof_epoch)+'_best.pkl'))
print('load success')

#%%
dm = np.load("sim_data/chengdu_directed_shortest_distance_matrix.npy")

def get_length(points_id, solution):
    length = 0
    end = len(solution) - 1
    for i in range(end):
        length += dm[points_id[solution[i]], points_id[solution[i+1]]]
    # # return point[0]
    # length += dm[points_id[solution[0]], points_id[solution[end]]]
    return length

def get_length_2(point, solution):
    length = 0
    end = len(solution)
    for i in range(end):
        length += math.sqrt((point[solution[i], 0] - point[i, 0]) ** 2
                            + (point[solution[i], 1] - point[i, 1]) ** 2)
    return length

#%%
model.eval()

# 读取测试数据
test_dataset = np.load('sim_data/chengdu_data.npy', allow_pickle=True)

test_dataloader = DataLoader(test_dataset,
                             batch_size=params.batch_size,
                             shuffle=True,
                             num_workers=0)

#%% 放置model到device
if USE_CUDA:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
else:
    model.to(device)

#%% 定义CrossEntropyLoss()和Adam优化器，并初始化losses
CCE = torch.nn.CrossEntropyLoss()
losses = []
batch_loss = []
iterator = tqdm(test_dataloader, unit='Batch')

start_time = time.time()
length_list = []
length_opt_list = []
solutions_list = []
solutions_opt_list = []
points_id_list = []
error_sum = 0
for i_batch, sample_batched in enumerate(iterator):
    test_batch_id = Variable(sample_batched['Points_id'])
    test_batch = Variable(sample_batched['Points'].float())
    target_batch = Variable(sample_batched['Solutions'])

    if USE_CUDA:
        test_batch_id = test_batch_id.cuda()
        test_batch = test_batch.cuda()
        target_batch = target_batch.cuda()
    else:
        test_batch_id = test_batch_id.to(device)
        test_batch = test_batch.to(device)
        target_batch = target_batch.to(device)

    o, p = model(test_batch)

    solutions = np.array(p.cpu())
    # print(solutions)
    points_id = np.array(test_batch_id.cpu())
    points = np.array(test_batch.cpu())
    solutions_opt = np.array(target_batch.cpu())

    error = 0

    for i in range(len(solutions)):
        length = get_length(points_id[i], solutions[i])
        # length = get_length_2(points[i], solutions[i])
        length_opt = get_length(points_id[i], solutions_opt[i])
        length_list.append(length)
        length_opt_list.append(length_opt)
        error_opt = (length - length_opt) / length_opt * 100
        error += error_opt
        solutions_list.append(solutions[i])
        solutions_opt_list.append(solutions_opt[i])
        points_id_list.append(points_id[i])

    error = error / len(solutions)
    error_sum += error

    o = o.contiguous().view(-1, o.size()[-1])
    target_batch = target_batch.view(-1)

    loss = CCE(o, target_batch)
    losses.append(loss.data.item())

end_time = time.time()
print('time: %.2f' % (end_time - start_time))
error_print = error_sum / len(iterator)
# print('current error: %.2f%%' % error)
print('average error: %.2f%%' % error_print)
print('current length: %.2f' % (sum(length_list) / len(length_list)))
print('current length_opt: %.2f' % (sum(length_opt_list) / len(length_opt_list)))

iterator.set_postfix(loss=np.average(losses))
print(np.average(losses))

#%% 对比选点
count = 0
for i in range(len(solutions_list)):
    points_id = points_id_list[i]
    for j in range(1, 5):
        if points_id[solutions_list[i][j]] == points_id[solutions_opt_list[i][j]]:
            count += 1
        # if points_id[solutions_list[i][j]] == points_id[solutions_opt_list[i][j]] and solutions_list[i][j] != solutions_opt_list[i][j]:
        #     print(solutions_list[i][j], solutions_opt_list[i][j])
print((count / (len(solutions_list) * 4)) * 100)

# #%% 对比选点
# count = 0
# for i in range(len(solutions_list)):
#     points_id = points_id_list[i]
#     for j in range(1, 5):
#         if solutions_list[i][j] == solutions_opt_list[i][j]:
#             count += 1
#         if points_id[solutions_list[i][j]] == points_id[solutions_opt_list[i][j]] and solutions_list[i][j] != solutions_opt_list[i][j]:
#             print(solutions_list[i][j], solutions_opt_list[i][j])
# print((count / (len(solutions_list) * 4)) * 100)

# #%% 对比顺序
# count = 0
# for i in range(len(solutions_list)):
#     flag = 0
#     for j in range(1, 5):
#         if int(solutions_list[i][j]/5) != int(solutions_opt_list[i][j]/5):
#             flag = 1
#             break
#     if flag == 0:
#         count += 1
# print((count / len(solutions_list)) * 100)

#%% 对比顺序+选点
count = 0
count_2 = 0
for i in range(len(solutions_list)):
    points_id = points_id_list[i]
    flag = 0
    for j in range(1, 5):
        if int(solutions_list[i][j]/5) != int(solutions_opt_list[i][j]/5):
            flag = 1
            break
    if flag == 0:
        count += 1
        flag_2 = 0
        for j in range(1, 5):
            if points_id[solutions_list[i][j]] != points_id[solutions_opt_list[i][j]]:
                flag_2 = 1
                break
        if flag_2 == 0:
            count_2 += 1
print((count / len(solutions_list)) * 100)
print((count_2 / len(solutions_list)) * 100)

