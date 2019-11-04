import pandas as pd
import scipy.sparse as sp
import numpy as np
import pickle
from collections import defaultdict


raw_allx = pd.read_csv('ind.poker.allx.csv', sep=',', header=None)
allx = sp.csr_matrix(raw_allx.values)

raw_tx = pd.read_csv('ind.poker.tx.csv', sep=',', header=None)
tx = sp.csr_matrix(raw_tx.values)

raw_x = pd.read_csv('ind.poker.x.csv', sep=',', header=None)
x = sp.csr_matrix(raw_x.values)


trainset_size = raw_allx.shape[0]
unlabelset_size = trainset_size - raw_x.shape[0]
testset_size = raw_tx.shape[0]

print("trainset_size: %d" % trainset_size)
print("unlabelset_size: %d" % unlabelset_size)
print("testset_size: %d" % testset_size)

raw_ally = pd.read_csv('ind.poker.ally.csv', sep=',', header=None)
raw_ally.columns = ['tag']


ally = []
for i in range(trainset_size):
    one_hot = [0 for l in range(2)]
    label_index = raw_ally.iat[i, 0]
    if label_index > -1:
        one_hot[label_index] = 1
    ally.append(one_hot)
ally = np.array(ally)


raw_y =pd.read_csv('ind.poker.y.csv', sep=',', header=None)
raw_y.columns = ['tag']
y = []
for i in range(trainset_size-unlabelset_size):
    one_hot = [0 for l in range(2)]
    label_index = raw_y.iat[i, 0]
    if label_index > -1 :
        one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)

raw_ty = pd.read_csv('ind.poker.ty.csv', sep=',', header=None)
raw_ty.columns = ['tag']
ty = []
for i in range(testset_size):
    one_hot = [0 for l in range(2)]
    label_index = raw_ty.iat[i, 0]
    if label_index > -1:
        one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)


raw_graph = pd.read_csv('ind.poker.graph.csv', sep=',', header=None)
graph = defaultdict(list)
for i in range(raw_graph.shape[0]):
    graph[raw_graph.iat[i,0]].append(raw_graph.iat[i,1])


# names = [(x,'x'), (y,'y'), (tx,'tx'), (ty,'ty'), (allx,'allx'), (ally,'ally'), (graph,'graph')]
# for i in range(len(names)):
#     with open('../data/ind.game.{}'.format(names[i][1]),'wb') as f:
#         pickle.dump(names[i][0], f, pickle.HIGHEST_PROTOCOL)


with open('../data/ind.poker.x','wb') as f:
    pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
with open('../data/ind.poker.y','wb') as f:
    pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)
with open('../data/ind.poker.tx','wb') as f:
    pickle.dump(tx, f, pickle.HIGHEST_PROTOCOL)
with open('../data/ind.poker.ty','wb') as f:
    pickle.dump(ty, f, pickle.HIGHEST_PROTOCOL)
with open('../data/ind.poker.allx','wb') as f:
    pickle.dump(allx, f, pickle.HIGHEST_PROTOCOL)
with open('../data/ind.poker.ally','wb') as f:
    pickle.dump(ally, f, pickle.HIGHEST_PROTOCOL)
with open('../data/ind.poker.graph','wb') as f:
    pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
