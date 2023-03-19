
from abc import ABC, abstractmethod


class BaseSymbolExtractor(ABC):
    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer

    def ingest_one_symbol_info(self, symbol_dict):
        self.feature_engineer.ingest_one_symbol_info(symbol_dict)

    @abstractmethod
    def train(self):
        raise NotImplemented()

    @abstractmethod
    @property
    def symbols(self):
        raise NotImplemented()

    def convert_string_to_X(self, string, max_edit_distance_considered=1):
        return self.feature_engineer.convert_inputstring_to_X(string, max_edit_distance_considered=max_edit_distance_considered)

    @abstractmethod
    def predict_proba(self, string, max_edit_distancce_considered=1):
        raise NotImplemented()

    @abstractmethod
    def save_model(self, directory):
        raise NotImplemented()

    @abstractmethod
    @classmethod
    def load_model(cls, directory, feature_engineer=None):
        raise NotImplemented()
