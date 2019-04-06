import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict

this_dir = osp.dirname(__file__)
project_root = osp.abspath( osp.join(this_dir, '..') )

cfg = edict()
cfg.cache_path = osp.join(project_root, '.cache')
cfg.output_path = osp.join(project_root, 'outputs')
cfg.embedding_size = 1024 # embedding dimensionality
cfg.seed = 1618
cfg.question_vocab_path = osp.join(project_root, 'data', 'question.vocab.json') # a joint question vocab across all dataset

# preprocess config
cfg.image_size = 448
cfg.output_size = cfg.image_size // 32
cfg.output_features = 2048
cfg.central_fraction = 0.875

# Train params
cfg.TRAIN = edict()

cfg.TRAIN.epochs = 50
cfg.TRAIN.batch_size = 128
cfg.TRAIN.base_lr  = 1e-3  # default Adam lr
cfg.TRAIN.lr_decay = 15    # in epochs
# cfg.TRAIN.data_workers = 20
cfg.TRAIN.data_workers = 10

cfg.TRAIN.answer_batch_size   = 3000 # batch size for answer network
cfg.TRAIN.max_negative_answer = 8000 # max negative answers to sample

# Test params
cfg.TEST = edict()
cfg.TEST.max_answer_index = 3000 # max answer index for computing acc

# Dataset params

# VQA2 params
cfg.VQA2 = edict()

cfg.VQA2.qa_path = osp.join(project_root, 'data', 'vqa2')
cfg.VQA2.feature_path = osp.join(project_root, 'features', 'vqa-resnet-14x14.h5')
cfg.VQA2.answer_vocab_path = osp.join(project_root, 'data', 'answer.vocab.vqa.json')
cfg.VQA2.train_img_path = osp.join(cfg.VQA2.qa_path, 'images', 'train2014')
cfg.VQA2.val_img_path   = osp.join(cfg.VQA2.qa_path, 'images', 'val2014')
cfg.VQA2.test_img_path  = osp.join(cfg.VQA2.qa_path, 'images', 'test-dev2015')

cfg.VQA2.train_qa = 'train2014'
cfg.VQA2.val_qa = 'val2014'
cfg.VQA2.test_qa = 'test-dev2015'

cfg.VQA2.task = 'OpenEnded'
cfg.VQA2.dataset = 'mscoco'

# VG params
cfg.VG = edict()

cfg.VG.qa_path = osp.join(project_root, 'data', 'vg')
cfg.VG.feature_path = osp.join(project_root, 'features', 'vg-resnet-14x14.h5')
cfg.VG.answer_vocab_path = osp.join(project_root, 'data', 'answer.vocab.vg.json')

cfg.VG.train_qa = 'VG_train_decoys.json'
cfg.VG.val_qa = 'VG_val_decoys.json'
cfg.VG.test_qa = 'VG_test_decoys.json'

cfg.VG.img_path = osp.join(cfg.VG.qa_path, 'images')

# V7W params
cfg.Visual7W = edict()

cfg.Visual7W.qa_path = osp.join(project_root, 'data', 'v7w')
cfg.Visual7W.feature_path = osp.join(project_root, 'features', 'vg-resnet-14x14.h5')
cfg.Visual7W.answer_vocab_path = osp.join(project_root, 'data', 'answer.vocab.v7w.json')

cfg.Visual7W.train_qa = 'v7w_train_questions.json'
cfg.Visual7W.val_qa   = 'v7w_val_questions.json'
cfg.Visual7W.test_qa  = 'v7w_test_questions.json'

#################################################################################
# A curated dataset for V7W, which removes bias towards modality
#   - See [Chao et. al. NAACL-HTL 2018 Being Negative but Constructively...] for
# details. We refer this dataset as V7W throughout the paper.
#################################################################################

cfg.Visual7W.train_v7w_decoys = 'v7w_train_decoys.json'
cfg.Visual7W.val_v7w_decoys   = 'v7w_val_decoys.json'
cfg.Visual7W.test_v7w_decoys  = 'v7w_test_decoys.json'

cfg.Visual7W.img_path = osp.join(cfg.Visual7W.qa_path, 'images')

def set_random_seed(seed):
  random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)

def update_train_configs(_cfg):
  cfg.TRAIN.batch_size          = _cfg.batch_size
  cfg.TRAIN.answer_batch_size   = _cfg.answer_batch_size
  cfg.TRAIN.max_negative_answer = _cfg.max_negative_answer

  if hasattr(_cfg, 'learning_rate'): # Handle optional attributes
    cfg.TRAIN.base_lr = _cfg.learning_rate

  if hasattr(_cfg, 'max_answer_index'):
    cfg.TEST.max_answer_index = _cfg.max_answer_index

