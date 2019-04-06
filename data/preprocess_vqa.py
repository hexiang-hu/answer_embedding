import sys
import os.path as osp
import json
import re
from collections import Counter, defaultdict, OrderedDict
from tqdm import tqdm
from IPython import embed

this_dir = osp.dirname(__file__)

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap    = {  'none': '0',
                  'zero': '0',
                  'one': '1',
                  'two': '2',
                  'three': '3',
                  'four': '4',
                  'five': '5',
                  'six': '6',
                  'seven': '7',
                  'eight': '8',
                  'nine': '9',
                  'ten': '10'
                }
articles     = ['a',
                'an',
                'the'
              ]

periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip   = re.compile("(\d)(\,)(\d)")
punct        = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def processPunctuation(inText):
  outText = inText
  for p in punct:
    if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
      outText = outText.replace(p, '')
    else:
      outText = outText.replace(p, ' ')
  outText = periodStrip.sub("", outText, re.UNICODE)

  return outText

def processDigitArticle(inText):
  outText = []
  tempText = inText.lower().split()
  for word in tempText:
    word = manualMap.setdefault(word, word)
    if word not in articles:
      outText.append(word)
    else:
      pass
  for wordId, word in enumerate(outText):
    if word in contractions:
      outText[wordId] = contractions[word]
  outText = ' '.join(outText)
  return outText

def preprocess_answer(answer):
  answer = answer.replace('\n', ' ')
  answer = answer.replace('\t', ' ')
  answer = answer.strip()
  answer = processPunctuation(answer)
  answer = processDigitArticle(answer)

  return answer

train_apath = osp.join(this_dir, 'vqa2', 'v2_mscoco_train2014_annotations.json')
val_apath   = osp.join(this_dir, 'vqa2', 'v2_mscoco_val2014_annotations.json')

train_qpath = osp.join(this_dir, 'vqa2', 'v2_OpenEnded_mscoco_train2014_questions.json')
val_qpath   = osp.join(this_dir, 'vqa2', 'v2_OpenEnded_mscoco_val2014_questions.json')

output_jsons = { 'train': 'vqa_train_questions.json', 'val': 'vqa_val_questions.json', 'test': 'vqa_test_questions.json' }
answer_conf  = { 'maybe': 0.5, 'yes': 1 , 'no': 0.1}
qa_pairs = { 'train': [], 'val': [] }

print('loading train json.')
with open(train_apath, 'r') as afile: train_anno = json.load(afile)['annotations']
with open(train_qpath, 'r') as qfile: train_ques = json.load(qfile)['questions']
assert len(train_anno) == len(train_ques)

print('loading val json.')
with open(val_apath, 'r') as afile: val_anno = json.load(afile)['annotations']
with open(val_qpath, 'r') as qfile: val_ques = json.load(qfile)['questions']
assert len(val_anno) == len(val_ques)

print('parsing train json.')
for q, a in tqdm(zip( train_ques, train_anno )):
  assert q['question_id'] == a['question_id']
  qa_pairs['train'].extend([{
    'qa_id': q['question_id'], 'question': q['question'], 'answers': [ preprocess_answer(_a['answer']) for _a in a['answers']], 'multiple_choice_answer': preprocess_answer(a['multiple_choice_answer']),
    'question_type': a['question_type'], 'image_index': a['image_id'], 'filename': 'COCO_train2014_{:012}.jpg'.format(a['image_id'])
    }])

print('parsing val json.')
for q, a in tqdm(zip( val_ques, val_anno )):
  assert q['question_id'] == a['question_id']
  qa_pairs['val'].extend([{
    'qa_id': q['question_id'], 'question': q['question'], 'answers': [ preprocess_answer(_a['answer']) for _a in a['answers']], 'multiple_choice_answer': preprocess_answer(a['multiple_choice_answer']),
    'question_type': a['question_type'], 'image_index': a['image_id'], 'filename': 'COCO_val2014_{:012}.jpg'.format(a['image_id'])
    }])

print('writing out parsed jsons.')
for k, v in qa_pairs.items():
  with open(osp.join(this_dir, 'vqa2', output_jsons[k]), 'w') as _file: json.dump(v, _file)
