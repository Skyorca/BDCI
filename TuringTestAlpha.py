'''
ver2.0.0 for Model Combination:
Model1: LDA
Model2: SVM< tf-idf, doc_topic_list>
MOdel3:......
'''

import csv
import sys
import pickle
import jieba.posseg as psg
import numpy as np
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm

def createPunc():
    punc = []
    fpunc = open("/home/skyfish/BDCI2017-360/punctuation.txt",'r')
    for line in fpunc:
        word = line.strip('\r\n')
        word = word.decode('utf-8')
        punc.append(word)
    return punc

def dataProcessForLDA(filename, trainNum):
    csv.field_size_limit(sys.maxsize)
    paperList = []
    counter = 0
    reader = csv.reader(filename , delimiter='\t')
    for line in reader:
        newLine = []
        if len(line[1])<100 & len(line[2])<1000:
            for i in range(4):
                newLine.append(line[i].decode('utf-8'))
        if len(newLine) > 0: paperList.append(newLine)
        counter += 1
        if counter % 5000 == 0: print 'doing...'
    #use paperList and re-organize data ready for IDA
    punc = createPunc()
    #posPick = ['f', 'c', 'p', 'uj', 'b', 'e', 'r', 'wp', 'ws', 'x','m', 'u','d','eng']
    posPick = ['a','n','nr','ns','nt','nz','t','v']
    docList = []
    for docNum in range(trainNum):
        newDoc  = []
        seg = psg.cut(paperList[docNum][2])
        for elem in seg:
            if(elem.word not in punc) & (elem.flag in posPick): newDoc.append(elem.word)
        newDoc.append(paperList[docNum][3]) # add POSITIVE or NEGATIVE
        docList.append(newDoc)
    #embedding into 2 matrix:tf_N & tf_P
    docList_Negative = []
    docList_Positive = []
    for docNum in range(len(docList)):
        if   docList[docNum][-1] == 'NEGATIVE':
            docList_Negative.append(" ".join(docList[docNum][:-1])) # note it's ' ' not  '' (with space), thus it's word by word
        elif docList[docNum][-1] == 'POSITIVE':
            docList_Positive.append(" ".join(docList[docNum][:-1]))

    tf_vectorizer_N = CountVectorizer(max_df=0.90, min_df=4)
    tf_vectorizer_P = CountVectorizer(max_df=0.90, min_df=4)
    tf_N = tf_vectorizer_N.fit_transform(docList_Negative)
    tf_P = tf_vectorizer_P.fit_transform(docList_Positive)
    #pickle.dump(tf_vectorizer_N, open('/home/skyfish/BDCI2017-360/tf_vectorizer_N.txt', 'w'))
    #pickle.dump(tf_vectorizer_P, open('/home/skyfish/BDCI2017-360/tf_vectorizer_P.txt', 'w'))
    pickle.dump(tf_N, open('/home/skyfish/BDCI2017-360/tf_N.txt', 'w'))
    pickle.dump(tf_P, open('/home/skyfish/BDCI2017-360/tf_P.txt', 'w'))

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])


def calcTopNum(topicList): # note that it's a numpy array
    counter = 1
    firstMaxElem = max(topicList)
    topicList[np.argmax(topicList)] = 0.000000001 # EQUAL TO REMOVE
    for i in range(len(topicList)):
        nextMaxElem = max(topicList)
        if (float(firstMaxElem/nextMaxElem)>0.) & (float(firstMaxElem/nextMaxElem)<500.):
            counter += 1
            topicList[np.argmax(topicList)] = 0.000000001
        else: continue
    return counter

def calcEntropy(topicList, n_topic):
    valueDict = []
    entropy = 0.0
    for elem in topicList:
        if elem not in valueDict.keys(): valueDict[elem] = 0
        valueDict[elem] += 1
    for elem in valueDict:
        prob = float(valueDict[elem])/n_topic
        entropy -= prob*log(prob, 2)
    return entropy

def ldaTest(label, trainNum, iter_time, n_top_words, filename): #label is a string
    fd = open('/home/skyfish/BDCI2017-360/train.tsv', 'r')
    dataProcessForLDA(fd, trainNum)
    if   label == 'NEGATIVE':
        #tf_vectorizer = pickle.load(open('/home/skyfish/BDCI2017-360/tf_vectorizer_N.txt', 'r'))
        tf            = pickle.load(open('/home/skyfish/BDCI2017-360/tf_N.txt', 'r'))
    elif label == 'POSITIVE':
        #tf_vectorizer = pickle.load(open('/home/skyfish/BDCI2017-360/tf_vectorizer_P.txt', 'r'))
        tf            = pickle.load(open('/home/skyfish/BDCI2017-360/tf_P.txt', 'r'))
    print 'doing LDA for %s...' % label
    parameters = {'learning_method':('batch', 'online'),
                  'n_components':range(20, 200, 5),
                  'perp_tol': (0.001, 0.01, 0.1),
                  'doc_topic_prior':(0.001, 0.01, 0.05, 0.1, 0.2),
                  'topic_word_prior':(0.001, 0.01, 0.05, 0.1, 0.2),
                  'max_iter':(200, 300, 5)}
    lda_model = LatentDirichletAllocation()
    lda = GridSearchCV(lda_model, parameters)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    #print_top_words(lda, tf_feature_names, n_top_words)
    doc_topic_list = lda.transform(tf)
    #print 'doc_topic_list_%s:' % label, doc_topic_list

    print 'calculating the divergence of each doc:'
    res = []
    for docNum in range(len(doc_topic_list)):
        #divergence = calcTopNum(doc_topic_list[docNum])
        divergence = calcEntropy(doc_topic_list[docNum], n_topic)
        res.append(divergence)
        filename.write('%s Doc %d has divergence %d \t' % (label, docNum, divergence))
    print 'divergence for %s is:' % label, np.mean(np.array(res))
    print 'perplexity = ', lda.perplexity(tf)
    fd.close()


def SVMTest(Method, filename, trainNum, n_topic, iter_time):
    csv.field_size_limit(sys.maxsize)
    paperList = []
    counter = 0
    reader = csv.reader(open('/home/skyfish/BDCI2017-360/train.tsv', 'r'), delimiter='\t')
    for line in reader:
        newLine = []
        if len(line[1])<100 & len(line[2])<1000:
            for i in range(4):
                newLine.append(line[i].decode('utf-8'))
        if len(newLine) > 0: paperList.append(newLine)
        counter += 1
        if counter % 5000 == 0: print 'doing...'
    #use paperList and re-organize data ready for IDA
    punc = createPunc()
    posPick = ['a','n','nr','ns','nt','nz','t','v']
    oldDocList = []
    for docNum in range(trainNum):
        newDoc  = []
        seg = psg.cut(paperList[docNum][2])
        for elem in seg:
            if(elem.word not in punc) & (elem.flag in posPick): newDoc.append(elem.word)
        newDoc.append(paperList[docNum][3]) # add POSITIVE or NEGATIVE
        oldDocList.append(newDoc)
    newDocList = [] # store available doc
    label      = [] # store every label
    for docNum in range(len(oldDocList)):
            newDocList.append(" ".join(oldDocList[docNum][:-1])) # note it's ' ' not  '' (with space)
            label.append(oldDocList[docNum][-1])
    for i in range(trainNum):
        if label[i] == 'NEGATIVE': label[i]=0
        else:                      label[i]=1
    #why we need oldDocList & newDocList? because we want a 'str' not separate words
    tf_vectorizer = CountVectorizer(max_df=0.90, min_df=4)
    tf = tf_vectorizer.fit_transform(newDocList)
    if Method == 'tf-idf':
        tfidftrans = TfidfTransformer()
        tfidf = tfidftrans.fit_transform(tf)
        X_train, X_test, y_train, y_test = train_test_split(tfidf, label, test_size = 0.20, random_state = 33)
        parameters = {'kernel':('linear','rbf'), 'C':[1,10]}
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(X_train, y_train)
        print 'The accuracy of tf-idf is :', clf.score(X_test, y_test)
    elif Method == 'theme-doc':
        lda = LatentDirichletAllocation(n_components=n_topic, max_iter=iter_time, learning_method='batch')
        lda.fit(tf)
        doc_topic_list = lda.transform(tf)
        X_train, X_test, y_train, y_test = train_test_split(doc_topic_list, label, test_size = 0.20, random_state = 33)
        parameters = {'kernel':('linear','rbf'), 'C':[1,10]}
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(X_train, y_train)
        print 'The accuracy of theme-doc is :', clf.score(X_test, y_test)
        print 'perplexity = ', lda.perplexity(tf)
'''
# for SVM
#SVMTest('tf-idf', 1000, 250, 300)
SVMTest('theme-doc', 1000, 400, 300)
'''
#for IDA
fn = open('/home/skyfish/BDCI2017-360/NegativeTopicDivergence.txt', 'w')
fp = open('/home/skyfish/BDCI2017-360/PositiveTopicDivergence.txt', 'w')
ldaTest('NEGATIVE', 500, 300, 15, fn)
ldaTest('POSITIVE', 500, 300, 15, fp)
fn.close()
fp.close()
