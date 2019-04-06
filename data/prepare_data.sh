#!/usr/bin/env bash
mkdir -p v7w
if [ -f "v7w/v7w_train_questions.json" ]; then
  echo "visual7w dataset exists, skipping..."
else
  echo "Download visual7w"
  wget -q "http://web.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip" -O v7w/dataset.zip
  unzip -j v7w/dataset.zip -d v7w/
  python preprocess_v7w.py
fi

mkdir -p vg
if [ -f "vg/VG_train_decoys.json" ]; then
  echo "qaVG dataset exists, skipping..."
else
  echo "Download qaVG"
  wget -q "http://hexianghu.com/vqa-negative-decoys/Visual_Genome_decoys.zip" -O vg/Visual_Genome_decoys.zip
  unzip -j vg/Visual_Genome_decoys.zip -d vg/
fi

mkdir -p vqa2
if [ -f "vqa2/v2_OpenEnded_mscoco_train2014_questions.json" ]; then
  echo "vqa2 dataset exists, skipping..."
else
  echo "Download vqa2"
  wget -q "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip" -O vqa2/v2_Questions_Train_mscoco.zip
  unzip -j vqa2/v2_Questions_Train_mscoco.zip -d vqa2/
  wget -q "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" -O vqa2/v2_Questions_Val_mscoco.zip
  unzip -j vqa2/v2_Questions_Val_mscoco.zip -d vqa2/
  wget -q "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip" -O vqa2/v2_Questions_Test_mscoco.zip
  unzip -j vqa2/v2_Questions_Test_mscoco.zip -d vqa2/
  wget -q "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" -O vqa2/v2_Annotations_Train_mscoco.zip
  unzip -j vqa2/v2_Annotations_Train_mscoco.zip -d vqa2/
  wget -q "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" -O vqa2/v2_Annotations_Val_mscoco.zip
  unzip -j vqa2/v2_Annotations_Val_mscoco.zip -d vqa2/
fi

python preprocess_vqa.py
