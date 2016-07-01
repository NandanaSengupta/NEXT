"""
BR_Random app implements DuelingBanditsPureExplorationPrototype
author: Kevin Jamieson, kevin.g.jamieson@gmail.com
last updated: 11/4/2015

BR_Random implements random sampling using the Borda reduction described in detail in
Jamieson et al "Sparse Borda Bandits," AISTATS 2015.
"""

import numpy as np
import numpy.random
import next.utils as utils
import logging


def getLossOnePair(w,i,j):
    #assumes i beat j
    if i==j:
        return 0
    else:
        return np.log(1. + np.exp(-w[i]+w[j]))

def getLoss(votes,w):
    loss = 0
    for vote in votes:
        loss += getLossOnePair(w,vote[0],vote[1])
    return loss


def epochSGD(w,votes,max_num_passes=0,a0=0.1,verbose=False):
    m = len(votes)
    n = len(w)
    if max_num_passes==0:
        max_iters = 16*m
    else:
        max_iters = max_num_passes*m

    epoch_length = m
    t = 0
    a = a0
    t_e = 0
    while t<max_iters:
        t += 1
        t_e += 1

        if t_e % epoch_length == 0:
            a = a*0.5
            epoch_length = 2*epoch_length
            t_e = 0

            #  if verbose:
                #  logging.debug("iter=%d, loss=%f, a=%f, error=%f" %(t,getLoss(votes,w),a,rankingDistance(w)))

        # select random pair
        x = np.random.randint(m)
        [idx,idy] = [votes[x][0], votes[x][1]]

        # sumeet: modify this
        temp = np.zeros(n)
        temp[idx] = -1.
        temp[idy] = +1.
        grad_partial = float(1)/(1.+np.exp(-w[idx]+w[idy])) * np.exp(-w[idx]+w[idy]) * temp
        w = w - a*grad_partial

    return w

def getGradient(votes, w):
    n = len(w)
    gd = np.zeros(n)
    m = len(votes)

    for count in range(m):
        [idx,idy] = [votes[count][0], votes[count][1]]

        temp = float(1)/(1.+np.exp(-w[idx]+w[idy])) * np.exp(-w[idx]+w[idy])
        gd[idx] += -temp
        gd[idy] += temp

    return gd

def gd(w, votes, max_iters=0, c1=0.0001, rho=0.5, verbose=False):
    """
    (numpy.array) C: count matrix of the training dataset
    (numpy.array) f: features matrix (n x d)
    (int) max_iters: maximum number of iterations of GD
    (float) c1: Amarijo stopping condition parameter (default 0.0001)
    (float) rho: Backtracking line search parameter (default 0.5)
    """
    if max_iters == 0:
        max_iters = 20

    alpha = 0.5
    t = 0
    loss_0 = float('inf')

    while t < max_iters:
        t += 1

        g = getGradient(votes, w)

        # backtracking line search
        alpha = 2 * alpha
        loss_0 = getLoss(votes, w)
        w_try = w-alpha*g
        loss_1 = getLoss(votes, w_try)

        inner_t = 0
        while loss_1 > loss_0 - c1 * alpha * np.linalg.norm(g):
            alpha = alpha * rho
            w_try = w-alpha*g
            loss_1 = getLoss(votes, w_try)
            inner_t += 1
            if inner_t > 100:
                return w, loss_0
        w = w_try

        #  if verbose:
            #  logging.debug("gd_iter=%d, loss=%f, a=%f, error=%f, inner_t=%f" %(t, loss_1, alpha, rankingDistance(w), inner_t))

    return w, loss_1

def optimization(w, votes, num_random_restarts=0, max_num_passes_SGD=0, max_iters_GD=0, verbose=False):
    if max_num_passes_SGD==0:
        max_num_passes_SGD = 16
    else:
        max_num_passes_SGD = max_num_passes_SGD

    if max_iters_GD == 0:
        max_iters_GD = 20
    else:
        max_iters_GD = max_iters_GD

    loss_old = float('inf')
    num_restarts = -1

    while num_restarts < num_random_restarts:
        num_restarts += 1

        w = epochSGD(w, votes, max_num_passes=max_num_passes_SGD, a0=0.1, verbose=verbose)

        w_new, loss_new = gd(w, votes, max_iters=max_iters_GD, verbose=verbose)

        if loss_new < loss_old:
            w_old = w_new
            loss_old = loss_new

        #  if verbose:
            #  logging.debug("restart %d: loss = %f, error=%f\n" %(num_restarts, loss_new, rankingDistance(w_new)))

    return w_old, loss_old

class UncertaintySampling:
        
    app_id = 'DuelingFeatures'
    def initExp(self, butler, n=None, failure_probability=None, features=None, params=None):
        """
        This function is meant to set keys used later by the algorith implemented
        in this file.
        """
        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='failure_probability', value=failure_probability)
        butler.algorithms.set(key='score', value=[0]*n)
        butler.algorithms.set(key='votes', value=[])
        butler.algorithms.set(key='votes', value=[0]*n)

        arm_key_value_dict = {}
        for i in range(n):
            arm_key_value_dict['Xsum_'+str(i)] = 0.
            arm_key_value_dict['T_'+str(i)] = 0.

        arm_key_value_dict.update({'total_pulls':0})

        butler.algorithms.set(key='keys', value=list(arm_key_value_dict.keys()))
        butler.algorithms.set_many(key_value_dict=arm_key_value_dict)

        return True

    def getQuery(self, butler, participant_uid, features):
        w = butler.algorithms.get(key='score')
        n = butler.algorithms.get(key='n')

        pairs = []
        differences = []
        sort_indices = np.argsort(w)
        minvalue = float("inf")
        for count in range(n-1):
            i = sort_indices[count]
            j = sort_indices[count+1]
            pairs.append([i,j])
            differences.append(abs(w[i]-w[j]))
        # TODO: Sumeet needs to fix this division
        # We experienced division by 0 when float(1) / x, changed it to
        # float(1) / (x + 1)
        probs = np.array([float(1)/(x + 1) for x in differences])
        probs = probs/np.sum(probs)
        cumprobs = np.cumsum(probs)
        p = np.random.random()
        index = np.where(cumprobs > p)[0][0]
        return pairs[index]

    def processAnswer(self,butler, left_id=0, right_id=0, winner_id=0,
                      features=None):

        votes = butler.algorithms.get(key='votes')
        if left_id==winner_id:
            votes.append([left_id,right_id])
        else:
            votes.append([right_id,left_id])
        butler.algorithms.set(key='votes', value=votes)

        n = butler.algorithms.get(key='n')
        w = np.random.random(n)
        score, loss = optimization(w, votes, num_random_restarts=0, max_num_passes_SGD=64, max_iters_GD=20, verbose=True)
        butler.algorithms.set(key='score', value=score)
        return True

    def getModel(self, butler):
        score = butler.algorithms.get(key='score')
        score = np.array(score)
        return (-np.sort(-score)).tolist(), [0]*len(score)
