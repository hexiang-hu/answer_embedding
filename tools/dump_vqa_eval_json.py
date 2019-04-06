import os, sys
import json
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True, type=str)
parser.add_argument('--iter', default=-1, type=int)
parser.add_argument('--input_json', default='data/vqa2/v2_OpenEnded_mscoco_val2014_questions.json', type=str)
parser.add_argument('--test_json',  default='data/vqa2/v2_OpenEnded_mscoco_test2015_questions.json', type=str)
parser.add_argument('--output_path', required=True, type=str)
args = parser.parse_args()

def invert_dict(d):
  return {v: k for k, v in d.items()}

def main(args):
  data = torch.load(args.input_path)
  evaluation = data['eval']
  if isinstance( evaluation, list ): evaluation = evaluation[args.iter]

  answer_ids   = torch.cat(evaluation['answer_ids']).numpy().tolist()
  question_ids = torch.cat(evaluation['question_ids']).numpy().tolist()
  print( 'Total number of answer: {}'.format( len(answer_ids) ) )

  with open(args.input_json, 'r') as fd: questions = json.load(fd)['questions']
  assert len(answer_ids) == len(questions)

  invert_vocab = data['vocab']['index_to_answer']

  diff_qids = None
  if 'test' in args.input_json:
    with open(args.test_json, 'r') as fd: test_questions = json.load(fd)['questions']
    dev_qids  = set( que['question_id'] for que in questions)
    test_qids = set( que['question_id'] for que in test_questions)
    diff_qids = test_qids - dev_qids

  with open(args.output_path, 'w') as fd:
    val_json = [ { 'answer': invert_vocab[answer], 'question_id': questions[idx]['question_id'] }
                    for answer, idx in zip(answer_ids, question_ids) ]

    if diff_qids is not None:
      val_json.extend([ {'question_id': qid, 'answer': 'yes' } for qid in diff_qids ])

    json.dump(val_json, fd)

if __name__ == '__main__':
  print(args.__dict__)
  main(args)
