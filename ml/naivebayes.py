
import os
import json

from sklearn.naive_bayes import MultinomialNB
import joblib

from ..symutils.featureengine import SymbolInfoFeatureEngineer
from .utils import BaseSymbolExtractor


class SymbolMultinomialNaiveBayesExtractor(BaseSymbolExtractor):
    def __init__(self, feature_engineer, alpha=1.):
        super(SymbolMultinomialNaiveBayesExtractor, self).__init__(feature_engineer)
        self.alpha = alpha
        self.symbols_weights_info = self.feature_engineer.symbols_weights_info

    def train(self):
        X, Y = self.feature_engineer.compute_training_data()
        self.classifier = MultinomialNB()
        self.classifier.fit(X, Y)

    @property
    def symbols(self):
        try:
            return list(self.classifier.classes_)
        except Exception:
            raise ValueError('Classifier not trained yet!')

    def predict_proba(self, string, max_edit_distance_considered=1):
        inputX = self.convert_string_to_X(string, max_edit_distance_considered=max_edit_distance_considered)
        proba = self.classifier.predict_proba(inputX)
        return {
            symbol: prob
            for symbol, prob in zip(self.symbols, proba[0, :])
        }

    def save_model(self, directory):
        self.feature_engineer.save_model(directory)
        json.dump({'alpha': self.alpha}, open(os.path.join(directory, 'hyperparameters.json'), 'w'))
        json.dump(self.symbols, open(os.path.join(directory, 'symbols.json'), 'w'))
        joblib.dump(self.classifier, os.path.join(directory, 'multinomialnb.joblib'))

    @classmethod
    def load_model(cls, directory, feature_engineer=None):
        if feature_engineer is None:
            feature_engineer = SymbolInfoFeatureEngineer.load_model(directory)
        hyperparameters = json.load(open(os.path.join(directory, 'hyperparameters.json'), 'r'))
        mclf = cls(feature_engineer, **hyperparameters)
        mclf.classifier = joblib.load(os.path.join(directory, 'multinomialnb.joblib'))
        return mclf
