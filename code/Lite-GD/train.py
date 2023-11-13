#%%
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

# 添加路径
import sys
sys.path.append('sim_data/')
from PointerNet import PointerNet
from Data_Generator import TSPDataset
from config import get_config

dm = np.load("sim_data/chengdu_directed_shortest_distance_matrix.npy")

def get_length(points_id, solution):
    length = 0
    end = len(solution) - 1
    for i in range(end):
        length += dm[points_id[solution[i]], points_id[solution[i+1]]]
    # # return point[0]
    # length += dm[points_id[solution[end]], points_id[solution[0]]]
    return length

if __name__ == '__main__':
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

    #%% test mode
    if params.test_flag == True:
        print('Test Mode!')
        model.load_state_dict(torch.load('code/Lite-GD/param/param_'+str(params.nof_points)+'_'+str(params.nof_epoch)+'.pkl'))
        print('load success')
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

        for i_batch, sample_batched in enumerate(iterator):
            test_batch = Variable(sample_batched['Points'].float())
            target_batch = Variable(sample_batched['Solutions'])

            if USE_CUDA:
                test_batch = test_batch.cuda()
                target_batch = target_batch.cuda()
            else:
                test_batch = test_batch.to(device)
                target_batch = target_batch.to(device)

            o, p = model(test_batch)
            o = o.contiguous().view(-1, o.size()[-1])
            target_batch = target_batch.view(-1)

            loss = CCE(o, target_batch)
            losses.append(loss.data.item())

        iterator.set_postfix(loss=np.average(losses))
        print(np.average(losses))

    #%% train mode
    else:
        print('Train mode!')
        model.load_state_dict(torch.load('code/Lite-GD/param/param_' + str(params.nof_points) + '_' + str(params.nof_epoch) + '.pkl'))
        print('load success')

        min_loss = 100
        #%% 读取training dataset
        train_dataset = np.load('sim_data/chengdu_data.npy', allow_pickle=True)

        dataloader = DataLoader(train_dataset,
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

        # 定义CrossEntropyLoss()和Adam优化器，并初始化losses
        CCE = torch.nn.CrossEntropyLoss()
        model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=params.lr)
        losses = []

        #%% Training process
        for epoch in range(params.nof_epoch):
            batch_loss = []
            iterator = tqdm(dataloader, unit='Batch')

            length_list = []
            length_opt_list = []
            error_sum = 0

            for i_batch, sample_batched in enumerate(iterator):
                iterator.set_description('Epoch %i/%i' % (epoch+1, params.nof_epoch))

                train_batch_id = Variable(sample_batched['Points_id'])
                train_batch = Variable(sample_batched['Points'].float())
                target_batch = Variable(sample_batched['Solutions'])
                target_batch_length = Variable(sample_batched['Opt_Length'].float())

                # 放置data到device
                if USE_CUDA:
                    train_batch_id = train_batch_id.cuda()
                    train_batch = train_batch.cuda()
                    target_batch = target_batch.cuda()
                    target_batch_length = target_batch_length.cuda()
                else:
                    train_batch_id = train_batch_id.to(device)
                    train_batch = train_batch.to(device)
                    target_batch = target_batch.to(device)
                    target_batch_length = target_batch_length.to(device)

                o, p = model(train_batch)

                solutions = np.array(p.cpu())
                points_id = np.array(train_batch_id.cpu())
                points = np.array(train_batch.cpu())
                solutions_opt = np.array(target_batch.cpu())
                opt_length = np.array(target_batch_length.cpu())

                error = 0

                for i in range(len(solutions)):
                    length = get_length(points_id[i], solutions[i])
                    # length = get_length_2(points[i], solutions[i])
                    # length_opt = get_length(points_id[i], solutions_opt[i])
                    length_opt = opt_length[i]
                    length_list.append(length)
                    length_opt_list.append(length_opt)
                    error_opt = (length - length_opt) / length_opt * 100
                    error += error_opt

                error = error / len(solutions)
                error_sum += error

                o = o.contiguous().view(-1, o.size()[-1])

                target_batch = target_batch.view(-1)

                loss = CCE(o, target_batch)
                losses.append(loss.data.item())
                batch_loss.append(loss.data.item())

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                # 更新进度条
                iterator.set_postfix(loss='{}'.format(loss.data.item()))

            print(solutions[0])
            print(solutions_opt[0])
            error_print = error_sum / len(iterator)
            # print('current error: %.2f%%' % error)
            print('average error: %.2f%%' % error_print)
            print('current length: %.2f' % (sum(length_list) / len(length_list)))
            print('current length_opt: %.2f' % (sum(length_opt_list) / len(length_opt_list)))
            # print('length of list: %.2f' % len(length_list))

            if min_loss > sum(batch_loss)/len(batch_loss):
                min_loss = sum(batch_loss)/len(batch_loss)
                torch.save(model.state_dict(), 'code/Lite-GD/param/param_'+str(params.nof_points)+'_'+str(params.nof_epoch)+'_best.pkl')

            # 更新进度条
            iterator.set_postfix(loss=sum(batch_loss)/len(batch_loss))

        #%% 存储模型
        torch.save(model.state_dict(), 'code/Lite-GD/param/param_'+str(params.nof_points)+'_'+str(params.nof_epoch)+'.pkl')
        print('save success')