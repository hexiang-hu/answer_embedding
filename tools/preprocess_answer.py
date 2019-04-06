import json
from collections import Counter
import itertools
import nltk
import argparse

import _init_paths
from ansemb.config import cfg
import ansemb.utils as utils
import ansemb.dataset.preprocess as prepro
from ansemb.vector import Vector

parser = argparse.ArgumentParser()
parser.add_argument('--max_answers', default=None, type=int)
parser.add_argument('--dataset', default='v7w', choices=['v7w', 'vqa', 'vg', 'vizwiz'])
parser.add_argument('--vocab_json_path', default='data/answer.vocab.{}.json', type=str)
args = parser.parse_args()

# import dataset specific dataloader
exec('import ansemb.dataset.{} as data'.format(args.dataset))

def parse_input(args):
  if args.dataset == 'v7w' or args.dataset == 'vg':
    train_questions = data.path_for(train=True)
    val_questions   = data.path_for(val=True)
    test_questions  = data.path_for(test=True)

    with open(train_questions, 'r') as fd:
      annotations = json.load(fd)
    with open(val_questions, 'r') as fd:
      annotations.extend( json.load(fd) )
    with open(test_questions, 'r') as fd:
      annotations.extend( json.load(fd) )
  elif args.dataset == 'vqa':
    train_answers = data.path_for(train=True, answer=True)
    val_answers   = data.path_for(val=True, answer=True)

    with open(train_answers, 'r') as fd:
      annotations = json.load(fd)['annotations']
    with open(val_answers, 'r') as fd:
      annotations.extend( json.load(fd)['annotations'] )
  elif args.dataset == 'vizwiz':
    train_answers = data.path_for(train=True)
    val_answers   = data.path_for(val=True)
    test_answers  = data.path_for(test=True)
    with open(train_answers, 'r') as fd:
      annotations = json.load(fd)['annotations']
    with open(val_answers, 'r') as fd:
      annotations.extend( json.load(fd)['annotations'] )
    with open(test_answers, 'r') as fd:
      annotations.extend( json.load(fd)['annotations'] )
  else:
    raise ValueError('Unsupported Dataset')

  return annotations

def main(args):
  output_format = args.vocab_json_path.format

  # process input json files
  annotations = parse_input(args)

  word2vec = Vector()

  answers = data.prepare_answers(annotations)
  answer_vocab = prepro.extract_vocab(answers, top_k=args.max_answers)

  vocabs = { 'answer': answer_vocab }
  print('* Dump output vocab to: {}'.format(output_format(args.dataset)))
  with open(output_format(args.dataset), 'w') as fd:
    json.dump(vocabs, fd)

if __name__ == '__main__':
  main(args)
