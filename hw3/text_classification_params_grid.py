from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
#import signal
import os
import json
import sys
import traceback


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker(object):
    def __init__(self):
        self.data = fetch_20newsgroups(
            subset='all', 
            categories=[
                'rec.autos',
                'rec.motorcycles',
                'rec.sport.baseball',
                'rec.sport.hockey'
            ], 
            remove=('headers', 'footers', 'quotes')
        )

        self.params = {
            "count_vectorizer_params": 
                {
                    "min_df": 0,
                    "max_df": 0.7,
                    "ngram_range": [1, 1]
                }, 
            "tfidf_transformer_params": 
                {
                    "norm": "l2"    
                }, 
            "logistic_regression_params": 
                {
                    "C": 14
                }
            }

    def check(self, params_path):
        try:
            with open(params_path, 'r') as f:
                params = self.params#json.load(f)

            #signal.signal(signal.SIGALRM, signal_handler)
            #signal.alarm(60)
            pipeline = make_pipeline(
                CountVectorizer(**params['count_vectorizer_params']), 
                TfidfTransformer(**params['tfidf_transformer_params']), 
                LogisticRegression(**params['logistic_regression_params'])
            )

            grid_params = {
                #'countvectorizer__ngram_range': [[1, 1], [1, 2], [1, 3], [1, 4]],
                #'countvectorizer__min_df': np.linspace(0, .5, 20),
                'countvectorizer__max_df': np.linspace(0.01, 1, 1000),
    
                #'logisticregression__C': [14],
                #'logisticregression__n_jobs': [-1],
                #'logisticregression__random_state': [7]
            }

            #score = np.mean(cross_val_score(
            #    pipeline, 
            #    self.data.data, 
            #    self.data.target,
            #    scoring='accuracy', 
            #    cv=3
            #))

            grid = GridSearchCV(pipeline, grid_params, scoring='accuracy', n_jobs=-1, cv=3, verbose=1)
            grid.fit(self.data.data, self.data.target)

        except:
            traceback.print_exception(*sys.exc_info())
            #score = None
        
        return grid.best_params_, grid.best_score_


if __name__ == '__main__':
    print(Checker().check(SCRIPT_DIR + '/text_classification_params_example.json'))