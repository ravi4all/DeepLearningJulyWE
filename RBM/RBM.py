import numpy as np
import pandas as pd
import torch

movies = pd.read_csv('ml-1m/movies.dat', header=None,
                     sep="::", engine='python',
                     encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', header=None,
                     sep="::", engine='python',
                     encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', header=None,
                     sep="::", engine='python',
                     encoding='latin-1')

trainingSet = pd.read_csv('ml-100k/u1.base',delimiter='\t')
trainingSet = np.array(trainingSet, dtype='int')

testSet = pd.read_csv('ml-100k/u1.test',delimiter='\t')
testSet = np.array(testSet, dtype='int')

nb_users = max(max(trainingSet[:,0]), max(testSet[:,0]))
nb_movies = max(max(trainingSet[:,1]), max(testSet[:,1]))

def convertData(data):
    newData = []
    
    for user_id in range(1, nb_users+1):
        movie_id = data[:,1][data[:,0] == user_id]
        movie_rating = data[:,2][data[:,0] == user_id]
        ratings = np.zeros(nb_movies)
        ratings[movie_id - 1] = movie_rating
        newData.append(list(ratings))
    
    return newData

trainingSet = convertData(trainingSet)
testSet = convertData(testSet)

trainingSet = torch.tensor(trainingSet)
testSet = torch.tensor(testSet)

trainingSet[trainingSet == 0] = -1
trainingSet[trainingSet == 1] = 0
trainingSet[trainingSet == 2] = 0
trainingSet[trainingSet > 3] = 1
testSet[testSet == 0] = -1
testSet[testSet == 1] = 0
testSet[testSet == 2] = 0
testSet[testSet >= 3] = 1


class RBM():
    
    def __init__(self, v, h):
        self.w = torch.randn(v,h)
        self.b1 = torch.randn(1,h)
        self.b2 = torch.randn(1,v)
    
    def sample_h(self, x):
#        1. matrix multiplication
#        2. Add bias
#        3. Activate using torch.sigmoid => p
#        4. return p, torch.bernoulli(p)
        pass
    
    def sample_v(self, y):
        pass
    
    def train(self):
        pass
    
    













