import json
import numpy
import numpy as np
import next.apps.SimpleTargetManager
import next.utils as utils


def rearrange_targets(targets, filenames):
    """
    targets: list of targets
    mapping: [(target filename, index), ..., (target filename, index)]

    We assume that the `target filename` is a filename AND that it is uploaded
    and this filename is present in the target URL.
    """
    new_targets = []
    for target in targets:
        target_present = [filename in target['primary_description'] for filename in filenames]
        if sum(target_present) != 1:
            raise Exception('Looks like multiple URLs had a filename present '
                            '(or not found in the list of URLs')

        new_targets += [targets[target_present.index(True)]]

    return new_targets


def get_features(butler):
    initExp_args = butler.experiment.get()['args']
    return initExp_args['features']

class DuelingFeatures(object):
    def __init__(self, db):
        self.app_id = 'DuelingFeatures'
        self.TargetManager = next.apps.SimpleTargetManager.SimpleTargetManager(db)

    def initExp(self, butler, exp_data):
        # TODO: change this in every app type coded thus far!
        if 'targetset' in exp_data['args']['targets'].keys():
            n = len(exp_data['args']['targets']['targetset'])
            if 'mapping' not in exp_data['args'].keys():
                raise Exception('When including a features targetset, '
                                'must also specify mapping')
            targetset = rearrange_targets(exp_data['args']['targets']['targetset'],
                                         exp_data['args']['mapping'])
            self.TargetManager.set_targetset(butler.exp_uid,
                                             targetset)
        else:
            n = exp_data['args']['targets']['n']
        exp_data['args']['n'] = n
        del exp_data['args']['targets']

        alg_data = {}
        algorithm_keys = ['n', 'failure_probability', 'features']
        for key in algorithm_keys:
            alg_data[key] = exp_data['args'][key]

        return exp_data, alg_data

    def getQuery(self, butler, alg, args):
        alg_response = alg({'participant_uid':args['participant_uid'],
                            'features': get_features(butler)})
        targets = [self.TargetManager.get_target_item(butler.exp_uid, alg_response[i])
                   for i in [0, 1, 2]]

        targets_list = [{'target':targets[0],'label':'left'}, 
                        {'target':targets[1],'label':'right'}]


        if targets[0]['target_id'] == targets[-1]['target_id']:
            targets_list[0]['flag'] = 1
            targets_list[1]['flag'] = 0
        else:
            targets_list[0]['flag'] = 0
            targets_list[1]['flag'] = 1

        return_dict = {'target_indices':targets_list}

        experiment_dict = butler.experiment.get()

        #if 'labels' in experiment_dict['args']['rating_scale']:
            #labels = experiment_dict['args']['rating_scale']['labels']
            #return_dict.update({'labels':labels})

        if 'context' in experiment_dict['args'] and 'context_type' in experiment_dict['args']:
            return_dict.update({'context':experiment_dict['args']['context'],'context_type':experiment_dict['args']['context_type']})

        return return_dict

    def processAnswer(self, butler, alg, args):
        query = butler.queries.get(uid=args['query_uid'])
        targets = query['target_indices']
        for target in targets:
            if target['label'] == 'left':
                left_id = target['target']['target_id']
            if target['label'] == 'right':
                right_id = target['target']['target_id']
            if target['flag'] == 1:
                painted_id = target['target']['target_id']

        winner_id = args['target_winner']
        butler.experiment.increment(key='num_reported_answers_for_' + query['alg_label'])

        alg({'left_id':left_id, 
             'right_id':right_id, 
             'winner_id':winner_id,
             'painted_id':painted_id,
             'features': get_features(butler)})
        return {'winner_id':winner_id}


    def getModel(self, butler, alg, args):
        scores, precisions = alg({'features': get_features(butler)})
        ranks = (-numpy.array(scores)).argsort().tolist()
        n = len(scores)
        indexes = numpy.array(range(n))[ranks]
        scores = numpy.array(scores)[ranks]
        precisions = numpy.array(precisions)[ranks]
        ranks = range(n)

        targets = []
        for index in range(n):
          targets.append( {'index':indexes[index],
                           'target':self.TargetManager.get_target_item(butler.exp_uid, indexes[index]),
                           'rank':ranks[index],
                           'score':scores[index],
                           'precision':precisions[index]} )
        num_reported_answers = butler.experiment.get('num_reported_answers')
        return {'targets': targets, 'num_reported_answers':num_reported_answers} 


