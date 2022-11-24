
import numpy as np
import sparse
from sklearn.naive_bayes import MultinomialNB
import numba as nb


def preprocess_symbol_tokens(symbol_dict):
    description = symbol_dict['description'].lower()
    symbol = symbol_dict['symbol'].lower()

    tokens = description.split(' ') + symbol.split(' ')
    return tokens


# reference: https://norvig.com/spell-correct.html
@nb.njit
def generate_edits1_tokens(word):
    letters = 'abcdefghijklmnopqrstuvwxyz1234567890/^'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def generate_edits2_tokens(word):
    return set(e2 for e1 in generate_edits1_tokens(word) for e2 in generate_edits1_tokens(e1))


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
            # for edits1_token in generate_edits1_tokens(token):
            #     token_weights[edits1_token] = self.gamma
            # for edits2_token in generate_edits2_tokens(token):
            #     token_weights[edits2_token] = self.gamma * self.gamma
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

    def predict_proba(self, string):
        tokens = string.lower().split(' ')
        nbfeatures = self.X.shape[1]
        inputX = np.zeros((1, nbfeatures))
        for token in tokens:
            inputX[0, self.feature2idx[token]] = 1.
        proba = self.classifier.predict_proba(inputX)
        return {
            self.symbols[i]: proba[0, i]
            for i in range(proba.shape[1])
        }
