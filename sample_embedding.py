import argparse
import sequence_modeling


def get_options(parser):
    parser.add_argument('-p', '--pad', action='store', nargs='?', const=None, default=30, type=int,
                        choices=None, help='Padding threshold. (default: 30)', metavar=None)
    parser.add_argument('-d', '--data', action='store', nargs='?', const=None, default="sst", type=str,
                        choices=None, help='Data to vectorize. (default: sst (StanfordSentimentTree))', metavar=None)
    parser.add_argument('-m', '--embed_model', action='store', nargs='?', const=None,
                        default="./data/GoogleNews-vectors-negative300.bin", type=str, choices=None, metavar=None,
                        help='Path to the embedding model. (default: ./data/GoogleNews-vectors-negative300.bin)')
    return parser.parse_args()


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)
    if args.data == "sst":
        data = sequence_modeling.sst("./data/stanfordSentimentTreebank", binary=True)
        sentences = data["sentence"]
        label = data["label"]
    else:
        raise ValueError("Unknown model !")

    sequence_modeling.vectorize_chunk(sentences=sentences, label=label, length=args.pad, chunk_size=5000,
                                      embed_path=args.embed_model, save_path="./data/embed_%i" % args.pad)





