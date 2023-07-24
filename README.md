# Lite-GD

- Environment
  - python 3.8
  - pytorch 1.12

- Sim data
  - R1-link.csv：original data for the Chengdu road network
  - chengdu_link_feature.txt: linkid, snode long, snode lat, enode long, enode lat, length, 0, 1
  - chengdu_node.txt：node id, longtitude, latitude, label
  - chengdu_link.txt：link id, snode, enode, length
  - chengdu_order_1000.txt: order id, edge list, ratio list
  - chengdu_label_1000.txt：order id, sequence, determined edge, node_path, edge_path, route_length
- code
  - Lite-GD: our proposed method
  - Disgreedy: greedy method based on the distance
  - Pointer Network: end-to-end supervised model
  - AM: combine RL with encoder-decoder framework
  - Graph2Route: the method for personalized takeaway delivery service
