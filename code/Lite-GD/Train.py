"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from PointerNet import PointerNet

import itertools
import argparse
import numpy as np

from scipy.spatial import distance_matrix
import torch
from tqdm import tqdm
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from haversine import haversine

#%%
def cal_acc(pred, label):
    for i in range(pred):
        if pred[i] != label[i]:
            return 0

def str2bool(str):
    return True if str.lower() == 'true' else False

def fix_geo_route(geo_list):
    waypoint_list = [0, 1, 2, 3, 4]
    # 构建candidate dict存储可能的length
    candidate_length_dict = {'01': [], '02': [], '12': [], '21': [], '13': [], '14': [], '23': [], '24': [],
                             '34': [],
                             '43': []}
    # 取出经纬度值
    point_list = []
    for geo in geo_list:
        point_list.append(geo)
    # 调用边最短路算法并存储入dict
    for key in candidate_length_dict.keys():
        node1 = point_list[waypoint_list.index(int(key[0:-1]))]
        node2 = point_list[waypoint_list.index(int(key[-1]))]
        candidate_length_dict[key].append(haversine(reversed(node1), reversed(node2)))

    # 根据4条可选路径计算最短route，输出长度及route类型
    candidate_route = [[0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [0, 1, 2, 4, 3], [0, 2, 1, 4, 3]]
    candidate_route_length = []
    for route in candidate_route:
        length = 0
        for i in range(0,4):
            length = length + candidate_length_dict[str(route[i]) + str(route[i + 1])][0]
        candidate_route_length.append(length)

    # 根据route类型输出route link_id序列
    route_type = candidate_route_length.index(min(candidate_route_length))
    route_type_list = candidate_route[route_type]

    # 打印接送驾顺序
    # print(route_type_list)

    return route_type_list

#%%
def main():
    parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

    # Data
    parser.add_argument('--train_size', default=10000, type=int, help='Training data size')
    parser.add_argument('--val_size', default=100, type=int, help='Validation data size')
    parser.add_argument('--test_size', default=100, type=int, help='Test data size')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    # Train
    parser.add_argument('--nof_epoch', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # GPU
    parser.add_argument('--gpu', default=False, action='store_true', help='Enable gpu')
    # TSP
    parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
    # Network
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
    parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout value')
    parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')
    # Test
    parser.add_argument('--test_flag', type=str2bool, default='True', help='To test')


    params = parser.parse_args()

    if params.gpu and torch.backends.mps.is_built():
        USE_MPS = True
    else:
        USE_MPS = False

    model = PointerNet(params.embedding_size,
                       params.hiddens,
                       params.nof_lstms,
                       params.dropout,
                       params.bidir)

    if params.test_flag:
        print('Loading model...')
        # model.load_state_dict(torch.load('param/parameter_geo_1667287761.5023758.pkl'))
        # model.load_state_dict(torch.load('param/parameter_1_1666894396.563787.pkl'))
        model.load_state_dict(torch.load('param/parameter_sd20.pkl'))
        # model.load_state_dict(torch.load('param/parameter_1_60.pkl'))
        # model.load_state_dict(torch.load('parameter.pkl'))
        print('Loaded finished!')

        def get_length(point, solution, ret_flag):
            length = 0
            end = len(solution) - 1
            for i in range(end):
                length += haversine(reversed(point[solution[i]]), reversed(point[solution[i+1]]))*1000
            if ret_flag == True:
                length += haversine(reversed(point[solution[end]]), reversed(point[solution[0]]))*1000

            return length

        test_point_num = 5

        dataset_test = np.load('data/test_sd.npy', allow_pickle=True)
        point_array = np.random.rand(len(dataset_test), test_point_num, 7)
        solution_geo = []
        for i in range(len(dataset_test)):
            point_array[i] = np.array(dataset_test[i]['Points'])
            solution_geo.append(fix_geo_route([x[0:2] for x in point_array[i]]))
        point_tensor = torch.tensor(point_array, dtype=torch.float)
        o, p = model(point_tensor)
        solutions = np.array(p)
        opt_list = []
        test_list = []
        geo_len_list = []
        error = []
        error_return = []
        count_dis_return = 0

        test_num = len(dataset_test)

        # show the result
        for i in range(test_num):
            point = point_array[i]
            solution = solutions[i]
            length = get_length([x[0:2] for x in point], solution, 0)  # test solution
            length_return = get_length([x[0:2] for x in point], solution, 1)

            solution_opt = np.array(dataset_test[i]['Solution'])
            length_opt = get_length([x[0:2] for x in point], solution_opt, 0)
            length_opt_return = get_length([x[0:2] for x in point], solution_opt, 1)

            length_geo = get_length([x[0:2] for x in point], solution_geo[i], 0)
            geo_len_list.append(length_geo)

            error_opt = (length - length_opt) / length_opt * 100
            error_return_opt = (length_return - length_opt_return) / length_opt_return * 100

            if error_return_opt <= 0:
                count_dis_return += 1

            opt_list.append(length_opt)
            test_list.append(length)
            error.append(error_opt)
            error_return.append(error_return_opt)

            # print('Test{0}:'.format(i + 1))
            # print(solution, 'length is ', length, '(Test solution)')
            # print(solution_opt, 'length is ', length_opt, '(Optimized solution)')
            # print('The length error is {0:.2f}%'.format(error_opt), '\n')

        # f = open('output/solution_exception.txt', 'w')
        count_dis = 0
        count_geo = 0
        count_acc = 0
        count_geo_opt = 0
        # count_pick = 0
        # count_delivery = 0
        # count_pick_delivery = 0
        for i in range(len(opt_list)):
            if (opt_list[i] >= test_list[i]):
                count_dis += 1
            if (geo_len_list[i] >= test_list[i]):
                count_geo += 1
            if (opt_list[i] == test_list[i]):
                count_acc += 1
            if (opt_list[i] <= geo_len_list[i]):
                count_geo_opt += 1
        #     else:
        #         if (solutions[i][1] != dataset_test[i]['Solution'][1]):
        #             count_pick += 1
        #         if (solutions[i][3] != dataset_test[i]['Solution'][3]):
        #             count_delivery += 1
        #         if (solutions[i][1] != dataset_test[i]['Solution'][1]) and (solutions[i][3] != dataset_test[i]['Solution'][3]):
        #             count_pick_delivery += 1
        # print(count_pick, count_delivery, count_pick_delivery)
        # print([x/len(dataset_test) for x in [count_pick, count_delivery, count_pick_delivery]])
        # print(len(dataset_test))
        # f.close()

        print('ACC rate for history: ', count_acc/len(opt_list) * 100, '%')
        print('Dis rate for history: ', count_dis/len(opt_list) * 100, '%')
        print('Dis rate for geo: ', count_geo/len(opt_list) * 100, '%')
        print('Geo ACC rate for history: ', count_geo_opt/len(opt_list) * 100, '%')
        # print('Dis return rate: ', count_dis_return / len(opt_list) * 100, '%')
        print('Error rate for history:', sum(error)/len(error), '%')
        # print('Error return rate:', sum(error_return) / len(error_return), '%')

        print('length reduction: ', sum(test_list)/sum(opt_list))
        print('length sum: ', sum(opt_list)/len(opt_list))

    else:
        dataset = np.load('data/train_sd.npy', allow_pickle=True)
        # dataset = dataset[0:int(len(dataset)/params.batch_size)*params.batch_size]
        # dataset: list of points and solutions(sequences)

        dataloader = DataLoader(dataset,
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=0)

        if USE_MPS:
            device = torch.device("mps")
            model = model.to(device)

        CCE = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
        model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=params.lr)

        filename = 'output/log_sd.txt'
        f = open(filename, 'w')

        losses = []
        for epoch in range(params.nof_epoch):
            batch_loss = []
            iterator = tqdm(dataloader, unit='Batch')
            for i_batch, sample_batched in enumerate(iterator):
                iterator.set_description('Epoch %i/%i' % (epoch + 1, params.nof_epoch))
                # solution_geo = []
                # for point_array in list(sample_batched['Points']):
                #     solution_geo.append(fix_geo_route([x[0:2] for x in point_array]))

                # print('geo solution: ', solution_geo, file = f)
                # print('historical solution: ', sample_batched['Solution'], file = f)

                if USE_MPS:
                    train_batch = Variable(sample_batched['Points'].to(device))
                    target_batch = Variable(sample_batched['Solution'].to(device))
                    # target_batch = Variable(torch.tensor(solution_geo)).to(device)
                else:
                    train_batch = Variable(sample_batched['Points'].float())
                    target_batch = Variable(sample_batched['Solution'])
                    # target_batch = Variable(torch.tensor(solution_geo))

                o, p = model(train_batch)

                o = o.contiguous().view(-1, o.size()[-1])

                target_batch = target_batch.view(-1)

                loss = CCE(o, target_batch)
                losses.append(loss.data)
                batch_loss.append(loss.data)

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                iterator.set_postfix(loss='{}'.format(loss.data))

            iterator.set_postfix(loss=sum(batch_loss)/len(batch_loss)) # np.average(batch_loss)
            print('Epoch ', epoch, ' loss: ', sum(batch_loss) / len(batch_loss))
            print('Epoch ', epoch, ' loss: ', sum(batch_loss)/len(batch_loss), file=f)

            if epoch % 20 == 19:
                torch.save(model.state_dict(), 'param/parameter_sd' + str(epoch + 1) + '.pkl')

        torch.save(model.state_dict(), 'param/parameter_sd' + f'{time.time()}' + '.pkl')
        print('save success')
        f.close()

if __name__ == '__main__':
    main()