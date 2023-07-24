import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv as GraphConv
from dgl.data import DGLDataset
import pandas as pd
import itertools
import numpy as np


# 计算node representation
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, h_feats, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# 计算edge representation
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2 + 7, h_feats)
        self.W2 = nn.Linear(h_feats, 4)

    def apply_edges(self, edges):
        h = torch.cat([edges.data['features'], edges.src["h"], edges.dst["h"]], 1)
        return {"representation": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["representation"]


# 返回所有case 边的表示
class FeedForwad(nn.Module):
    def __init__(self):
        super(FeedForwad, self).__init__()
        self.node_model = GCN(2, 16)
        self.edge_model = MLPPredictor(16)
        self.edge_rep = {}

    def forward(self, graphs):
        for i in range(len(graphs)):
            h = self.node_model(graphs[i], graphs[i].ndata["feat"])
            h_e = self.edge_model(graphs[i], h)
            self.edge_rep[i] = h_e
        return self.edge_rep


class Decoder(nn.Module):
    def __init__(self, link2edge, case_edges_features):
        super().__init__()
        self.output_labels = {}
        self.link2edge = link2edge
        self.case_edges_features = case_edges_features
        self.linear = nn.Linear(5941, 32)

        self.mask_on = Parameter(torch.ones(1), requires_grad=False)
        self.runner_on = Parameter(torch.zeros(1), requires_grad=False)
        self.mask_off = Parameter(torch.ones(1), requires_grad=False)
        self.runner_off = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, edge_rep):
        # 获取case中的id
        edges_group = self.case_edges_features.groupby("case_id")
        for case_id in edges_group.groups:

            shape_0 = case_cadidate_nums[case_id]
            shape_1 = candidate_length[case_id] - 1
            indices = torch.zeros((shape_1, shape_0))

            mask_on = self.mask_on.repeat(shape_0).unsqueeze(0) # 34，1

            # Generating arang(input_length), broadcasted across batch_size
            runner_on = self.runner_on.repeat(shape_0)
            for i in range(shape_0):
                runner_on.data[i] = i
            # runner_on = runner_on.unsqueeze(0).expand(shape_0, -1).long()
            runner_on = runner_on.unsqueeze(0).long()


            edges_of_id = edges_group.get_group(case_id).reset_index()
            # 当前case下，所有边的表示 case_rep
            case_rep = edge_rep[case_id]
            # 确定是2、3个乘客，则序列长度为5或7
            passenger_nums, seq_nums = (2, 5) if max(edges_of_id['is_driver']) == 5 else (3, 7)
            # 当前case下，其他边的表示
            passenger_ids = edges_of_id.loc[(edges_of_id['is_driver'] != 0) & (edges_of_id['is_driver'] != 1)]
            reset_passenger_ids = passenger_ids.reset_index()
            passenger_reps_list = [case_rep[self.link2edge[passenger_ids['edge_id'][i + 1]]] for i in
                                   range(len(passenger_ids))]
            passenger_distance_list = [distance_matrix[self.link2edge[passenger_ids['edge_id'][i + 1]]] for i in
                                   range(len(passenger_ids))]
            passenger_distance = torch.stack(passenger_distance_list)
            passenger_rep = torch.stack(passenger_reps_list)  # 所有乘客的全部上车、下车的表示
            passenger_reps = torch.concat([passenger_rep, self.linear(passenger_distance)], dim=1)
            # passenger_reps_off = torch.concat([passenger_rep, self.linear(passenger_distance)], dim=1)
            # papassenger_reps_off = passenger_reps_on

            tmp_ones = torch.ones_like(passenger_reps)

            driver_id = edges_of_id['edge_id'][0]
            driver_rep = case_rep[self.link2edge[driver_id]]
            # indices.append(driver_id)
            driver_rep = torch.concat([driver_rep, self.linear(distance_matrix[driver_id])])
            # 计算上车点
            mask = ((passenger_ids['is_driver'] == 2) | (passenger_ids['is_driver'] == 4) | (
                        passenger_ids['is_driver'] == 6)).to_numpy()
            for i in range(passenger_nums):
                not_mask = ~mask
                mask_on.squeeze()[not_mask] = 0
                # passenger_reps_on[not_mask] = 0
                driver_rep = driver_rep.unsqueeze(1)
                attn = torch.mm(passenger_reps, driver_rep)
                pred = attn.argmax().item()
                driver_id = reset_passenger_ids['edge_id'][pred]
                driver_rep = torch.concat([case_rep[self.link2edge[driver_id]], self.linear(distance_matrix[driver_id])])

                # 现在，只是在选择输出哪个id时，使用mask的信息，即：选择输出哪个id时，不考虑attn中下车点对应的结果
                # 下面，计算应该输出谁
                is_driver = reset_passenger_ids['is_driver'][pred]  # 标识是第几个乘客，2，4，6
                mask = (mask & (passenger_ids['is_driver'] != is_driver)).to_numpy()

                indices[i] = torch.nn.functional.softmax(attn, dim=0).squeeze()
                masked_attn = attn * mask_on.T
                # Get maximum probabilities and indices
                max_probs, indice = masked_attn.max(0)
                one_hot_pointers = (runner_on == indice.unsqueeze(1).expand(-1, shape_0)).float()

                # Update mask to ignore seen indices, 需要将同组的信息都mask掉
                mask_on = mask_on * (1 - one_hot_pointers)

            # 计算下车点
            # passenger_rep = torch.stack(passenger_reps_list)  # 所有乘客的全部上车、下车的表示
            # passenger_reps = torch.concat([passenger_rep, self.linear(passenger_distance)], dim=1)
            mask = ((passenger_ids['is_driver'] == 3) | (passenger_ids['is_driver'] == 5) | (
                        passenger_ids['is_driver'] == 7)).to_numpy()
            for i in range(passenger_nums):
                not_mask = ~mask
                # papassenger_reps_off[not_mask] = 0
                driver_rep = driver_rep.unsqueeze(1)
                attn = torch.mm(passenger_reps, driver_rep)
                attn = torch.mm(tmp_ones[not_mask], attn)

                pred = attn.argmax().item()
                driver_id = reset_passenger_ids['edge_id'][pred]
                # indices.append(driver_id)
                indices[i + passenger_nums] = torch.nn.functional.softmax(attn, dim=0).squeeze()
                is_driver = reset_passenger_ids['is_driver'][pred]  # 标识是第几个乘客，2，4，6
                mask = (mask & (passenger_ids['is_driver'] != is_driver)).to_numpy()
                # driver_rep = case_rep[self.link2edge[driver_id]]
                driver_rep = torch.concat([case_rep[self.link2edge[driver_id]], self.linear(distance_matrix[driver_id])])
            self.output_labels[case_id] = indices
        return self.output_labels


# 读取真实的候选点输出
candidate_labels = {}
candidate_length = {}
# 成都：1902个节点，5941条边
with open("/Users/mali/Documents/0-Research/Yucen-Lite-GD/code-0218/sim_data/chengdu_order_label.txt", 'r') as file:
    data = file.readlines()
    count = 0
    for i in range(len(data)):
        if i % 2 == 0:
            candidate_labels[count] = data[i+1].strip().split(',')         # 真实路径
            candidate_length[count] = len(data[i+1].strip().split(','))    # 长度用来标识是双拼（长度为5）还是三拼（长度为7）
        else:
            continue
        count = count + 1


# 1. 用R1-link构建graph
# 2. 用pretrain_test读取边特征
# 3. label=0, 是候选点，不是最优解；label=1, 不是候选点，不是最优解；label=2, 不是候选点，是最优解；label=3, 是候选点，是最优解
nodes_data = pd.read_csv("~/Documents/0-Research/Yucen-Lite-GD/code-0218/sim_data/chengdu_node.txt", sep=" ", header=None)
edges_data = pd.read_csv("~/Documents/0-Research/Yucen-Lite-GD/code-0218/sim_data/chengdu_link_feature.txt", sep=" ", header=None)
print(nodes_data.shape)
print(edges_data.shape)

print(nodes_data.head(3))
print(edges_data.head(3))

node_features = torch.from_numpy(nodes_data.iloc[:, 1:3].to_numpy()).float()  # longtitude latitude
edge_features = torch.from_numpy(edges_data.iloc[:, [2, 3, 5, 6, 7, 8, 9]].to_numpy()).float()  # length


distance_matrix = torch.load('./distance_matrix.pth')

edges_src = torch.from_numpy(edges_data.iloc[:, 1].to_numpy())
edges_dst = torch.from_numpy(edges_data.iloc[:, 4].to_numpy())

# 对每个graph来说，除了来自case_edges_features中的特征是需要更新的，剩下的边特征是默认的
column_names = list(
    "case_id,edge_id,Node_Start,Longitude_Start,Latitude_Start,Node_End,Longitude_End,Latitude_End,Length,is_driver,ratio,label".split(
        ","))
case_edges_features = pd.read_csv("~/Documents/0-Research/Yucen-Lite-GD/code-0218/sim_data/chengdu_case_feature.txt",
                                  sep=" ", header=None, names=column_names)

# 处理真实标签
true_labels = {}
case_cadidate_nums = {}
edges_group = case_edges_features.groupby("case_id")
for case_id in edges_group.groups:
    edges_of_id = edges_group.get_group(case_id)
    # 统计每个case有多少个候选点
    candidates = edges_of_id.loc[(edges_of_id['is_driver']!=0) & (edges_of_id['is_driver']!=1)].reset_index()
    candidate_nums = len(candidates)
    case_cadidate_nums[case_id] = candidate_nums
    edgeid2index = {}
    for i in range(candidate_nums):
        edgeid2index[candidates['edge_id'][i]] = i
    labels = []
    old_labels = candidate_labels[case_id]
    # 去掉司机的位置
    for i in range(1, candidate_length[case_id]):
        # 找到old_labels在candidates这个数组中真实对应的下标
        old_label = int(old_labels[i]) # 3693
        index = edgeid2index[old_label]
        labels.append(index)
    true_labels[case_id] = labels


node_nums = nodes_data.shape[0]
edge_nums = edges_data.shape[0]
case_nums = case_edges_features.iloc[:, 0].max()
link2edge = {}
for i in range(edge_nums):
    link2edge[edges_data.iloc[i, 0]] = i

default_g = dgl.graph((edges_src, edges_dst), )
default_g.ndata["feat"] = node_features

case_graphs = {}
case_labels = {}
edges_group = case_edges_features.groupby("case_id")
for case_id in edges_group.groups:
    edges_of_id = edges_group.get_group(case_id)
    default_labels = [1 for j in range(edge_nums)]
    default_g.edata["features"] = edge_features
    for i in range(len(edges_of_id)):
        # 取出dgl graph中实际对应的edge_id
        edge_id = link2edge[edges_of_id["edge_id"].to_numpy()[i]]
        default_g.edata["features"][edge_id] = torch.from_numpy(edges_of_id.iloc[0, [3, 4, 6, 7, 8, 9, 10]].to_numpy())
        default_labels[edge_id] = edges_of_id["label"].to_numpy()[i]
    case_graphs[case_id] = default_g
    case_labels[case_id] = torch.from_numpy(np.asarray(default_labels))

encoder = FeedForwad()
decoder = Decoder(link2edge, case_edges_features)
import itertools
optimizer = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.01
)
for epoch in range(100):
    # forward
    edge_rep = encoder(case_graphs)
    output_labels = decoder(edge_rep)
    total_loss = 0
    for k, v in output_labels.items():
        # v 是 [edge_num, 4]的二维向量
        # logits = v
        # pred = v.argmax(1)
        labels = true_labels[k]
        total_loss += F.cross_entropy(v, torch.tensor(labels))
    # backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print("In epoch {}, loss: {}".format(epoch, total_loss))
