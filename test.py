#!/usr/bin/env python
# coding=utf-8

#a series of tests for data-dealing

#test for words-division
'''
import jieba
import jieba.posseg

list = []
fp = open("/home/skyfish/BDCI2017-360/test.txt", "r")
fr = open("/home/skyfish/BDCI2017-360/test_result.txt", "w")
for i in fp.readlines():
    seg = (jieba.posseg.cut(i))
    for elem in seg:
        list.append(elem.encode('utf-8'))
    #list.append('\n')
    for elem in range(len(list)):
        if list[elem] != '\n':
            fr.write(list[elem]+',')
        else:
            fr.write(list[elem])
    list = []
fp.close()
fr.close()
'''


'''
#test for reading .tsv
import csv
import sys
import pickle
csv.field_size_limit(sys.maxsize)
paperList = []
counter = 0
reader = csv.reader(open('/home/skyfish/BDCI2017-360/train.tsv'), delimiter='\t')
for line in reader:
    newLine = []
    if len(line[1])<100 & len(line[2])<1000:
        for i in range(4):
            newLine.append(line[i].decode('utf-8'))
    paperList.append(newLine)
    counter += 1
    if counter % 5000 == 0: print 'doing...'
pickle.dump(paperList, open('/home/skyfish/BDCI2017-360/traindata_pickle.txt','w'))
'''



#test for all jieba/pickle
import jieba
import pickle
newlist = pickle.load(open('/home/skyfish/BDCI2017-360/traindata_pickle.txt','r'))
seg = jieba.cut(newlist[0][1])
for elem in seg:
    print elem
