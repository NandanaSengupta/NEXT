import numpy
import numpy.random
import next.utils as utils


class BR_Random:
    def initExp(self, butler, n=None, features=None, params=None):
        """
        This function is meant to set keys used later by the algorith
        implemented in this file.
        """
        butler.algorithms.set(key='n', value=n)

        arm_key_value_dict = {}
        # loop over every target and set valeus we need later
        for i in range(n):
            arm_key_value_dict['Xsum_'+str(i)] = 0.
            arm_key_value_dict['T_'+str(i)] = 0.

        arm_key_value_dict.update({'total_pulls': 0})

        butler.algorithms.set(key='keys',
                              value=list(arm_key_value_dict.keys()))
        butler.algorithms.set_many(key_value_dict=arm_key_value_dict)

        return True

    def getQuery(self, butler, participant_uid, features):
        n = butler.algorithms.get(key='n')

        index1 = numpy.random.choice(n)
        index2 = numpy.random.choice(n)
        while index1 == index2:
            index2 = numpy.random.choice(n)

        random_fork = numpy.random.choice(2)
        if random_fork == 0:
            return [index1, index2]

        return [index2, index1]

    def processAnswer(self, butler, left_id=0, right_id=0, winner_id=0,
                      features=None):
        for index in [left_id, right_id]:
            reward = 1.0 if winner_id == index else 0.0
            d = {'Xsum_' + str(winner_id): reward,
                 'T_' + str(winner_id): 1.0,
                 'total_pulls': 1}
            butler.algorithms.increment_many(key_value_dict=d)
        return True

    def getModel(self, butler):
        keys = butler.algorithms.get(key='keys')
        key_value_dict = butler.algorithms.get(key=keys)
        n = butler.algorithms.get(key='n')

        sumX = [key_value_dict['Xsum_'+str(i)] for i in range(n)]
        T = [key_value_dict['T_'+str(i)] for i in range(n)]

        mu = numpy.zeros(n, dtype='float')
        for i in range(n):
            if T[i] == 0 or mu[i] == float('inf'):
                # this happens when no user has responded with this and no
                # rating for the target is possible
                mu[i] = -1
            else:
                # this happens when the query has been presented
                mu[i] = sumX[i] * 1.0 / T[i]

        prec = [numpy.sqrt(1.0 / max(1, t)) for t in T]
        return mu.tolist(), prec
