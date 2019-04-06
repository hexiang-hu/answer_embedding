import sys
import os.path as osp
import json

this_dir = osp.dirname(__file__)
with open(osp.join(this_dir, 'v7w', 'dataset_v7w_telling.json'), 'r') as _file:
  qa_anno = json.load(_file)['images']

print 'parsing json.'
output_jsons = { 'train': 'v7w_train_questions.json', 'val': 'v7w_val_questions.json', 'test': 'v7w_test_questions.json' }
qa_pairs = { 'train': [], 'val': [], 'test': [] }
for qas in qa_anno:
  qa_pairs[qas['split']].extend( [ { 'qa_id': qa['qa_id'], 'question': qa['question'], 'answer': qa['answer'],
                                     'multiple_choices': qa['multiple_choices'] + [ qa['answer'] ], 'question_type': qa['type'],
                                     'image_index': qas['image_id'], 'filename': qas['filename'] } for qa in qas['qa_pairs']] )

print 'writing out parsed jsons.'
for k, v in qa_pairs.items():
  with open(osp.join(this_dir, 'v7w', output_jsons[k]), 'w') as _file: json.dump(v, _file)
