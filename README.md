# Answer Embedding
Code Release for [Learning Answer Embeddings for Visual Question Answering](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Learning_Answer_Embeddings_CVPR_2018_paper.pdf). (CVPR 2018)

## Usage


```
usage: train_v7w_embedding.py [-h] [--gpu_id GPU_ID] [--batch_size BATCH_SIZE]
                              [--max_negative_answer MAX_NEGATIVE_ANSWER]
                              [--answer_batch_size ANSWER_BATCH_SIZE]
                              [--loss_temperature LOSS_TEMPERATURE]
                              [--pretrained_model PRETRAINED_MODEL]
                              [--context_embedding {SAN,BoW}]
                              [--answer_embedding {BoW,RNN}] [--name NAME]

optional arguments:
  -h, --help            show this help message and exit
  --gpu_id GPU_ID
  --batch_size BATCH_SIZE
  --max_negative_answer MAX_NEGATIVE_ANSWER
  --answer_batch_size ANSWER_BATCH_SIZE
  --loss_temperature LOSS_TEMPERATURE
  --pretrained_model PRETRAINED_MODEL
  --context_embedding {SAN,BoW}
  --answer_embedding {BoW,RNN}
  --name NAME
```

## Bibtex

Please cite with the following bibtex if you are using any related resource of this repo for your research. 

```
@inproceedings{hu2018learning,
  title={Learning Answer Embeddings for Visual Question Answering},
  author={Hu, Hexiang and Chao, Wei-Lun and Sha, Fei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5428--5436},
  year={2018}
}
```

## Acknowledgement
Part of this code uses components from [pytorch-vqa](https://github.com/Cyanogenoid/pytorch-vqa) and [torchtext](https://github.com/pytorch/text). We thank authors for releasing their code. 

## References

1. Being Negative but Constructively:
Lessons Learnt from Creating Better Visual Question Answering Datasets ([qaVG website](http://www.teds.usc.edu/website_vqa/))
2. Visual7W: Grounded Question Answering in Images
 ([website](http://web.stanford.edu/~yukez/visual7w/index.html))
3. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering [website](http://www.visualqa.org/)
