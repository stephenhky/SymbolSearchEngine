
import os
import json
from itertools import product

import sparse
import jellyfish


def preprocess_symbol_tokens(symbol_dict):
    description = symbol_dict['description'].lower()
    symbol = symbol_dict['symbol'].lower()

    tokens = description.split(' ') + symbol.split(' ')
    return tokens


class SymbolInfoFeatureEngineer:
    def __init__(self, gamma=0.75):
        self.gamma = gamma
        self.symbols_weights_info = {}

    def ingest_one_symbol_info(self, symbol_dict):
        symbol = symbol_dict['symbol']
        token_weights = {}
        for token in preprocess_symbol_tokens(symbol_dict):
            token_weights[token] = 1.
        self.symbols_weights_info[symbol] = token_weights

    def _produce_feature_indices(self):
        feature_set = set()
        for token_weights in self.symbols_weights_info.values():
            for feature in token_weights.keys():
                feature_set.add(feature)
        self.feature2idx = {
            feature: idx
            for idx, feature in enumerate(sorted(feature_set))
        }
        self.idx2feature = {idx: feature for feature, idx in self.feature2idx.items()}

    def _construct_training_data(self):
        X = sparse.DOK((len(self.symbols_weights_info), len(self.feature2idx)))
        Y = []
        for i, (symbol, weights_info) in enumerate(self.symbols_weights_info.items()):
            Y.append(symbol)
            for feature, weight in weights_info.items():
                X[i, self.feature2idx[feature]] = weight
        X = X.to_coo().to_scipy_sparse().tocsr()
        return X, Y

    def compute_training_data(self):
        self._produce_feature_indices()
        return self._construct_training_data()

    def convert_inputstring_to_X(self, string, max_edit_distance_considered=1):
        tokens = string.lower().split(' ')
        nbfeatures = len(self.feature2idx)
        inputX = sparse.DOK((1, nbfeatures))
        for token in tokens:
            if token in self.feature2idx.keys():
                inputX[0, self.feature2idx[token]] = 1.
        if max_edit_distance_considered > 0:
            for token, feature in product(tokens, self.feature2idx.keys()):
                if token == feature:
                    continue
                edit_distance = jellyfish.damerau_levenshtein_distance(token, feature)
                if edit_distance <= max_edit_distance_considered:
                    inputX[0, self.feature2idx[feature]] = pow(self.gamma, edit_distance)
        return inputX.to_coo().to_scipy_sparse().tocsr()

    def save_model(self, directory):
        json.dump(self.feature2idx, open(os.path.join(directory, 'feature2idx.json'), 'w'))
        json.dump(self.symbols_weights_info, open(os.path.join(directory, 'symbols_weight_info.json'), 'w'))
        json.dump({'gamma': self.gamma}, open(os.path.join(directory, 'feature_engineer_hyperparameters.json'), 'w'))

    @classmethod
    def load_model(cls, directory):
        feature_engineer_hyperparameters = json.load(
            open(os.path.join(directory, 'feature_engineer_hyperparameters.json'), 'r')
        )
        feature_engineer = SymbolInfoFeatureEngineer(**feature_engineer_hyperparameters)
        feature_engineer.symbols_weights_info = json.load(open(os.path.join(directory, 'symbols_weight_info.json'), 'r'))
        feature_engineer.feature2idx = json.load(open(os.path.join(directory, 'feature2idx.json'), 'r'))
        feature_engineer.idx2feature = {idx: feature for feature, idx in feature_engineer.feature2idx.items()}
        return feature_engineer
