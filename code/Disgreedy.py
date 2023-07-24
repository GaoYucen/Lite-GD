#!/usr/bin/env python
# -*- coding:utf-8 -*-
#%%
import sys
import numpy
import math
import numpy as np
from tqdm import tqdm

#%%
totalnum = 0
sameorder = 0

num_stat = {3: 0, 5: 0, 7: 0}

file = open('20220420.txt', 'r')
for line in tqdm(file):
    line = line.strip().split("\t")
    order0, order1, order2, edlist, driver, gpoint, triplist = line[2], line[4], line[6], line[7], line[9], line[-2], \
                                                               line[-1]
    orderlist = triplist.split(",")
    if orderlist[0] != "-1": continue
    if len(orderlist) != 3: continue
    driver = driver.replace("(", "").replace(")", "").split(",")

    """
    gplist = []
    if gpoint != "NULL" :
        for gp in gpoint.split("|") :
            gpcols = gp.split(",")
            gplist.append(gpcols[1].replace("[","").replace("]",""))
    """
    # print driver
    distdict = {}
    for etadist in edlist.split(","):
        if etadist == "": continue
        ori, dst, eta, dist = etadist.split("|")
        key = ori + "_" + dst
        distdict[key] = float(dist.split(":")[1])

    totaldist = 0
    odlist = {}
    for idx in range(len(orderlist) - 1):
        key = "ori:" + orderlist[idx] + "_" + "dst:" + orderlist[idx + 1]
        totaldist += distdict[key]
        odlist.setdefault(orderlist[idx], {})
    odlist.setdefault(orderlist[-1], {})

    totalnum += 1
    orderdict = {}
    orderdict1 = {}
    if order0 != "NULL":
        order0 = order0.split(")to(")
        o0 = order0[0].replace("(", "").split(",")
        d0 = order0[1].replace(")", "").split(",")
        odlist["0"]["lng"] = float(o0[0])
        odlist["0"]["lat"] = float(o0[1])
        odlist["1"]["lng"] = float(d0[0])
        odlist["1"]["lat"] = float(d0[1])
        orderdict[0] = [abs(round(float(o0[0]) - float(driver[0]), 6)), abs(round(float(o0[1]) - float(driver[1]), 6))]
        orderdict[1] = [abs(round(float(d0[0]) - float(driver[0]), 6)), abs(round(float(d0[1]) - float(driver[1]), 6))]
    if order1 != "NULL":
        order1 = order1.split(")to(")
        o1 = order1[0].replace("(", "").split(",")
        d1 = order1[1].replace(")", "").split(",")
        odlist["2"]["lng"] = float(o1[0])
        odlist["2"]["lat"] = float(o1[1])
        odlist["3"]["lng"] = float(d1[0])
        odlist["3"]["lat"] = float(d1[1])
        orderdict[2] = [abs(round(float(o1[0]) - float(driver[0]), 6)), abs(round(float(o1[1]) - float(driver[1]), 6))]
        orderdict[3] = [abs(round(float(d1[0]) - float(driver[0]), 6)), abs(round(float(d1[1]) - float(driver[1]), 6))]
    if order2 != "NULL":
        order2 = order2.split(")to(")
        o2 = order2[0].replace("(", "").split(",")
        d2 = order2[1].replace(")", "").split(",")
        odlist["4"]["lng"] = float(o2[0])
        odlist["4"]["lat"] = float(o2[1])
        odlist["5"]["lng"] = float(d2[0])
        odlist["5"]["lat"] = float(d2[1])
        orderdict[4] = [abs(round(float(o2[0]) - float(driver[0]), 6)), abs(round(float(o2[1]) - float(driver[1]), 6))]
        orderdict[5] = [abs(round(float(d2[0]) - float(driver[0]), 6)), abs(round(float(d2[1]) - float(driver[1]), 6))]
    newlist = [str(-1)]
    """
    directiondict = {}
    for key,items in orderdict1.items() :
        newkey = 0 
        if items[0] > 0 and items[1] > 0 : #右上
            newkey = 0
        elif items[0] < 0 and items[1] > 0 : #左上
            newkey = 1
        elif items[0] < 0 and items[1] < 0 : #左下
            newkey = 2
        else : #右下
            newkey = 3
        directiondict.setdefault(key,{})
        #directiondict[newkey][key] = abs(items[0]) + abs(items[1])
        directiondict[key] = abs(items[0]) + abs(items[1])
    #print directiondict
    for key,items in sorted(directiondict.items(),key = lambda x : np.mean(x[1].values())) :
        #print key,items
        for it,dist in sorted(items.items(),key = lambda x : x[1]) :
            newlist.append(str(it))
    """
    # 先判断前2个接驾,如果第一个接驾接到，理论上不会去送第一个用户，而是去接第2个用户，否则属于送驾前接单了
    # print orderdict
    for key, items in sorted(orderdict.items(), key=lambda x: np.sum(x[1])):
        if key % 2 != 0: continue
        newlist.append(str(key))
        if len(newlist) >= 3:
            continue
    # print newlist
    # 判断和最后一个接驾的距离，先去送谁或接谁
    for key, items in orderdict.items():
        if str(key) in newlist: continue
        orderdict[key] = [abs(round(odlist[str(key)]["lng"] - odlist[newlist[-1]]["lng"], 6)),
                          abs(round(odlist[str(key)]["lat"] - odlist[newlist[-1]]["lat"], 6))]
        # 需要更新剩下的一个接驾和所有送驾
    for key, items in sorted(orderdict.items(), key=lambda x: np.sum(x[1])):
        if str(key) in newlist: continue
        newlist.append(str(key))

    if ",".join(newlist) != triplist:
        sameorder += 1
    # print totaldist,orderdict,orderdict1,",".join(newlist),triplist,line
print(totalnum, sameorder, sameorder * 1.0 / totalnum)

#%%
print(num_stat)
print(num_stat[3] + num_stat[5] + num_stat[7])
