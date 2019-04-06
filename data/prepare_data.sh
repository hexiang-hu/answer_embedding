#!/usr/bin/env bash
mkdir -p v7w
if [ -f "v7w/dataset_v7w_telling.json" ]; then
  echo "visual7w dataset exists, skipping..."
else
  echo "Download visual7w"
  wget -q "http://web.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip" -O v7w/dataset.zip
  unzip -j v7w/dataset.zip
fi

python preprocess_v7w.py

mkdir -p vg
if [ -f "vg/VG_train_decoys.json" ]; then
  echo "qaVG dataset exists, skipping..."
else
  echo "Download qaVG"
  wget -q "http://hexianghu.com/vqa-negative-decoys/Visual_Genome_decoys.zip"
  unzip -j vg/Visual_Genome_decoys.zip
fi

mkdir -p vqa2
if [ -f "vqa2/v2_OpenEnded_mscoco_train2014_questions.json" ]; then
  echo "vqa2 dataset exists, skipping..."
else
  echo "Download vqa2"
  wget -q "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip" -O vqa2/v2_Questions_Train_mscoco.zip
  unzip -j vqa2/v2_Questions_Train_mscoco.zip
  wget -q "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" -O vqa2/v2_Questions_Val_mscoco.zip
  unzip -j vqa2/v2_Questions_Val_mscoco.zip
  wget -q "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip" -O vqa2/v2_Questions_Test_mscoco.zip
  unzip -j vqa2/v2_Questions_Test_mscoco.zip
fi

python preprocess_vqa.py
