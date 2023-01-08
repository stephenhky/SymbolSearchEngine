
from argparse import ArgumentParser
import json
import os

from mnbutils import SymbolMultinomialNaiveBayesExtractor, SymbolInfoFeatureEngineer


def get_argparser():
    argparser = ArgumentParser(description='Train a multinomial naive Bayes model for symbol extraction')
    argparser.add_argument('symbolinfopath', help='path of all symbol information (*.json)')
    argparser.add_argument('modeldir', help='path of the directory for the model files')
    argparser.add_argument('--alpha', type=float, default=1., help='alpha for Laplace smoothing (default: 1.0)')
    argparser.add_argument('--gamma', type=float, default=0.75, help='gamma for decay matching (gamma: 0.75)')
    return argparser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    if not os.path.isdir(args.modeldir):
        os.makedirs(args.modeldir)
    symbolinfo = json.load(open(args.symbolinfopath, 'r'))

    feature_enginner = SymbolInfoFeatureEngineer(gamma=args.gamma)
    extractor = SymbolMultinomialNaiveBayesExtractor(feature_enginner, alpha=args.alpha)
    for info in symbolinfo:
        extractor.ingest_one_symbol_info(info)
    extractor.train()
    extractor.save_model(args.modeldir)
