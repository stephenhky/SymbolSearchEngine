
import numpy as np
import sparse
from sklearn.naive_bayes import MultinomialNB
import jellyfish


def preprocess_symbol_tokens(symbol_dict):
    description = symbol_dict['description'].lower()
    symbol = symbol_dict['symbol'].lower()

    tokens = description.split(' ') + symbol.split(' ')
    return tokens


class SymbolMultinomialNaiveBayesExtractor:
    def __init__(self, alpha=1., gamma=0.75):
        self.alpha = alpha
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
        self.X = sparse.DOK((len(self.symbols_weights_info), len(self.feature2idx)))
        self.Y = []
        for i, (symbol, weights_info) in enumerate(self.symbols_weights_info.items()):
            self.Y.append(symbol)
            for feature, weight in weights_info.items():
                self.X[i, self.feature2idx[feature]] = weight
        self.X = self.X.to_coo().to_scipy_sparse()

    def train(self):
        self._produce_feature_indices()
        self._construct_training_data()
        self.classifier = MultinomialNB()
        self.classifier.fit(self.X, self.Y)

    @property
    def symbols(self):
        try:
            return list(self.classifier.classes_)
        except Exception:
            raise ValueError('Classifier not trained yet!')

    def convert_string_to_X(self, string, max_edit_distance_considered=2):
        tokens = string.lower().split(' ')
        nbfeatures = self.X.shape[1]
        inputX = np.zeros((1, nbfeatures))
        for token in tokens:
            if token in self.feature2idx.keys():
                inputX[0, self.feature2idx[token]] = 1.
        if max_edit_distance_considered > 0:
            for token in tokens:
                for feature in self.feature2idx.keys():
                    if token == feature:
                        continue
                    edit_distance = jellyfish.damerau_levenshtein_distance(token, feature)
                    if edit_distance <= max_edit_distance_considered:
                        inputX[0, self.feature2idx[feature]] = pow(self.gamma, edit_distance)
        return inputX

    def predict_proba(self, string, max_edit_distance_considered=2):
        inputX = self.convert_string_to_X(string, max_edit_distance_considered=max_edit_distance_considered)
        proba = self.classifier.predict_proba(inputX)
        return {
            symbol: prob
            for symbol, prob in zip(self.symbols, proba[0, :])
        }
