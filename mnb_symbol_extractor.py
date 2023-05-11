from argparse import ArgumentParser

from symutils.ml.naivebayes import SymbolMultinomialNaiveBayesExtractor


def get_argparser():
    argparser = ArgumentParser(description='Extract symbol with query string')
    argparser.add_argument('modeldir', help='path for the model directory')
    argparser.add_argument('--topn', default=5, type=int, help='number of symbols to display (default: 5)')
    argparser.add_argument('--maxedit', default=1, type=int, help='maximum edit distance allowed (default: 1)')
    return argparser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    extractor = SymbolMultinomialNaiveBayesExtractor.load_model(args.modeldir)

    done = False
    while not done:
        querystring = input('Search> ')
        if len(querystring.strip()) <= 0:
            done = True
            continue
        ans = extractor.predict_proba(querystring, max_edit_distance_considered=args.maxedit)
        for symbol, proba in sorted(ans.items(), key=lambda item: item[1], reverse=True)[:args.topn]:
            print('{}\t{}'.format(symbol, proba))

    print('--DONE--')