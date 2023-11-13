#%% 读取R1-linK.csv
import pandas as pd
import numpy as np
import networkx as nx

# #%% 读取node和link
# data = pd.read_csv('sim_data/R1-link.csv')
# node = pd.read_csv('sim_data/chengdu_node.txt', sep = ',', header = None, names = ['Node', 'Longitude', 'Latitude'])
# link = pd.read_csv('sim_data/chengdu_link.txt', sep = ' ', header = None, names = ['Node_Start', 'Node_End', 'Length'])
#
# #%%
# for i in range(len(data)):
#     if data['Node_Start'].iloc[i] > 115:
#         data['Node_Start'].iloc[i] = data['Node_Start'].iloc[i] - 1
#     if data['Node_End'].iloc[i] > 115:
#         data['Node_End'].iloc[i] = data['Node_End'].iloc[i] - 1
#
# #%%
# for i in range(len(link)):
#     if link['Node_Start'].iloc[i] > 115:
#         link['Node_Start'].iloc[i] = link['Node_Start'].iloc[i] - 1
#     if link['Node_End'].iloc[i] > 115:
#         link['Node_End'].iloc[i] = link['Node_End'].iloc[i] - 1
#
# #%%
# for i in range(len(node)):
#     if node['Node'].iloc[i] > 115:
#         node['Node'].iloc[i] = node['Node'].iloc[i] - 1
#
# #%%
# data.to_csv('sim_data/R1-link-mod.txt', index = False, header = True, sep=',')
# link.to_csv('sim_data/chengdu_link-mod.txt', index = False, header = True, sep=',')
# node.to_csv('sim_data/chengdu_node-mod.txt', index = False, header = True, sep=',')

#%% 读取node和link
data = pd.read_csv('sim_data/R1-link-mod.txt', sep = ',', header = 0)
link = pd.read_csv('sim_data/chengdu_link-mod.txt', sep = ',', header = 0)
node = pd.read_csv('sim_data/chengdu_node-mod.txt', sep = ',', header = 0)


#%% 读图
G = nx.DiGraph()
# 添加边
for i in range(len(link)):
    G.add_edge(link['Node_Start'].iloc[i], link['Node_End'].iloc[i], weight = link['Length'].iloc[i])

#%% 检查有向边 159条有向边
count = 0
for e in G.edges():
    if (e[1], e[0]) not in G.edges():
        print(e)
        count += 1
print(count)

#%%
n = len(G)
print(n)

distance_matrix = np.zeros((n, n))

# vertex index
vertex_index = {}

i = 0
for v in G.nodes():
    vertex_index[v] = i
    i += 1

for s in G.nodes():
    length = nx.single_source_dijkstra_path_length(G, s)
    s_i = vertex_index[s]
    for t in length:
        t_i = vertex_index[t]
        distance_matrix[s_i][t_i] = length[t]

print(distance_matrix)
np.save("sim_data/chengdu_shortest_distance_matrix.npy", distance_matrix)

