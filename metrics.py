from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

from typing import Dict
from collections import defaultdict

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Metric(object):
    def __init__(self):
        self.result_container = defaultdict(list)

    def evaluate(self, y_true, y_pred) -> Dict:
        f1 = f1_score(y_true, y_pred, average='macro')
        pr = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        self.result_container['f1'].append(f1)
        self.result_container['pr'].append(pr)
        self.result_container['recall'].append(recall)

        return {
            'f1': f1,
            'pr': pr,
            'recall': recall,
        }

    def reset(self) -> None:
        self.result_container.clear()

    def save_mean_result(self, save_path='result.txt') -> Dict:
        result = {}
        for k, v in self.result_container.items():
            result[k] = np.mean(v)

        with open(save_path, 'w') as fout:
            for k, v in result.items():
                logger.info('%s: %10.4f' % (k, v))
                fout.write('%s: %10.4f\n' % (k, v))
        logger.info('save results at %s' % save_path)

        return result
