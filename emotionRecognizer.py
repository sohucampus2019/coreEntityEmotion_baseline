from sklearn.naive_bayes import MultinomialNB
from gensim import models, corpora

from scipy.sparse import csr_matrix
import numpy as np

import os
import json

import codecs

from typing import List, Dict

from data_reader import Dataset, build_dataset, build_vocabulary, tokenize_examples_, KeyMapping
from metrics import Metric

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

dir_prefix = 'data'
train_data_file = os.path.join(dir_prefix, 'train.txt')
test_data_file = os.path.join(dir_prefix, 'submit_entity_part.txt')
submit_data_file = os.path.join(dir_prefix, 'submit.txt')

label_mapping = {k: idx for idx, k in enumerate(Dataset.get_labels())}


def convert_examples_to_feature(voc: corpora.Dictionary, examples: List[Dict], window_size=50):
    tfidf_model = models.TfidfModel(dictionary=voc)

    # windows size left 50 and right 50 of aspect word
    def find_index(word_ls, target):
        for i, word in enumerate(word_ls):
            if word == target:
                return i
        return -1

    corpus = []
    labels = []
    for example in examples:
        text_tokens = example[KeyMapping.title] + example[KeyMapping.content]
        for item in example[KeyMapping.labels]:
            aspect_idx = find_index(text_tokens, item[KeyMapping.entity])
            labels.append(label_mapping[item[KeyMapping.polarity]])

            l_idx = aspect_idx - window_size if aspect_idx > window_size else 0
            r_idx = aspect_idx + window_size if aspect_idx + window_size <= len(text_tokens) else len(text_tokens)
            corpus.append(voc.doc2bow(text_tokens[l_idx:r_idx]))

    # gensim vector to sklearn form
    data = []
    rows = []
    cols = []

    voc_len = len(voc.token2id)
    for idx, doc in enumerate(corpus):
        for elem in tfidf_model[doc]:
            rows.append(idx)
            cols.append(elem[0])
            data.append(elem[1])

        # padding to voc_len
        rows.append(idx)
        cols.append(voc_len)
        data.append(0)

    tfidf_sparse_matrix = csr_matrix((data, (rows, cols)))
    tfidf_matrix = tfidf_sparse_matrix.toarray()
    return tfidf_matrix, np.array(labels)


def train(model, times=5, line_range=(0, None)):
    metric = Metric()
    dataset = build_dataset(train_data_file, line_range)
    tokenize_examples_(dataset)

    logger.info('build vocabulary')
    voc = build_vocabulary(dataset)
    voc.filter_extremes()

    def run_once():
        # split
        train_examples, test_examples = dataset.split_dataset(train_ratio=0.7)

        logger.info('split: train_size:%d, test_size:%d' % (len(train_examples), len(test_examples)))

        logger.info('convert docs to features and labels')
        train_features, train_labels = convert_examples_to_feature(voc, train_examples)
        test_features, test_labels = convert_examples_to_feature(voc, test_examples)

        logger.info('start training')
        model.fit(train_features, train_labels)

        # testing
        logger.info('start testing')

        for k, v in label_mapping.items():
            indices = np.nonzero(test_labels == v)[0]
            logger.info('# of {} class is {}'.format(k, len(indices)))

        y_pred = model.predict(test_features)
        # print and save result
        metric.evaluate(test_labels, y_pred)

    # run multi times
    for i in range(times):
        run_once()
    metric.save_mean_result()

    return model, voc


def test(model, voc):
    dataset = build_dataset(test_data_file)
    tokenize_examples_(dataset)
    # logger.info('build vocabulary')
    # voc.merge_with(build_vocabulary(dataset))

    reverse_label_mapping = {k: v for v, k in label_mapping.items()}
    for example in dataset.input_examples:
        features, _ = convert_examples_to_feature(voc, [example])  # fake label in example
        y_pred = model.predict(features)

        assert len(features) == len(example[KeyMapping.labels])
        for idx, item in enumerate(example[KeyMapping.labels]):
            item[KeyMapping.polarity] = reverse_label_mapping[y_pred[idx]]

    # write result to submit_file
    with codecs.open(submit_data_file, 'w', 'utf-8') as fout:
        for example in dataset.input_examples:
            example.pop(KeyMapping.title)
            example.pop(KeyMapping.content)
            fout.write(json.dumps(example, ensure_ascii=False) + '\n')

    logger.info('save submit file at {}'.format(submit_data_file))


if __name__ == '__main__':
    model, voc = train(model=MultinomialNB(), times=5, line_range=(0, None))
    test(model, voc)
