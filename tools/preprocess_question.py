import json
import itertools
import nltk
import argparse

import _init_paths
import ansemb.dataset.vqa as data_vqa
import ansemb.dataset.vg as data_vg
import ansemb.dataset.v7w as data_v7w
from ansemb.dataset.preprocess import extract_vocab
import ansemb.utils as utils
from ansemb.vector import Vector

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_json_path', default='data/question.vocab.json', type=str)
args = parser.parse_args()

def main(args):
  vqa_train_questions = data_vqa.path_for(train=True, question=True)
  vqa_val_questions   = data_vqa.path_for(val=True, question=True)

  vg_train_questions = data_vg.path_for(train=True)
  vg_val_questions   = data_vg.path_for(val=True)
  vg_test_questions  = data_vg.path_for(test=True)

  v7w_train_questions = data_v7w.path_for(train=True)
  v7w_val_questions   = data_v7w.path_for(val=True)
  v7w_test_questions  = data_v7w.path_for(test=True)

  with open(vqa_train_questions, 'r') as fd:
      vqaq = json.load(fd)['questions']
  with open(vqa_val_questions, 'r') as fd:
      vqaq.extend( json.load(fd)['questions'] )

  with open(vg_train_questions, 'r') as fd:
      vg = json.load(fd)
  with open(vg_val_questions, 'r') as fd:
      vg.extend( json.load(fd) )
  with open(vg_test_questions, 'r') as fd:
      vg.extend( json.load(fd))

  with open(v7w_train_questions, 'r') as fd:
      v7w = json.load(fd)
  with open(v7w_val_questions, 'r') as fd:
      v7w.extend( json.load(fd) )
  with open(v7w_test_questions, 'r') as fd:
      v7w.extend( json.load(fd) )

  word2vec = Vector()
  max_question_length = max( max(map(len, data_vqa.prepare_questions(vqaq))),
                             max(map(len, data_vg.prepare_questions(vg))),
                             max(map(len, data_v7w.prepare_questions(v7w))))

  questions = [ q for q in data_vqa.prepare_questions(vqaq) ]
  questions.extend([ q for q in data_vg.prepare_questions(vg) ])
  questions.extend([ q for q in data_v7w.prepare_questions(v7w) ])

  question_vocab  = extract_vocab(questions, start=1, input_vocab=word2vec)
  question_vocab['<UNK>'] = 0 # set token 0 as unknown token

  vocabs = {
      'question': question_vocab,
      'max_question_length': max_question_length,
  }
  with open(args.vocab_json_path, 'w') as fd:
      json.dump(vocabs, fd)

if __name__ == '__main__':
  main(args)
