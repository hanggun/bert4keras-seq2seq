# bert4keras-seq2seq
An implementation of attention is all you need using bert4keras

This repository gives three applications for two datasets: multi30k-en-de--machine translation dataset and CSL--title generation dataset

For multi30k-en-de dataset, we use BPE tokenization, English dataset and German dataset do not share the same embedding. For CSL chinese dataset, we use simplified BERT tokenization.

For machine translation dataset, we only use transformer architecture; For CSL NLG dataset, in addition to transformer, we also apply a BERT+Decoder architecture.

The result can be seen below:</br>
|model|dataset|rouge-1|rouge-2|rouge-l|bleu|
|:---:|:---:|:---:|:---:|:---:|:---:|
|seq2seq|multi30k-en-de-test2016|0.6834|0.4675|0.6553|0.3455|
|seq2seq|CSL-test|0.4366|0.2528|0.3853|0.1517|
|BERT+Deocder|CSL-test|0.6189|0.4430|0.5743|0.3326|

### Use bert4keras-seq2seq
- just run `python train.py` and `python evaluate_bpe_data.py` and you can train and evaluate a multi30k machine translation model. All the configuration is written in `yamls/multi30k.yaml`.

- For CSL dataset and seq2seq model, change the yaml path in `config.py` to `yamls/csl10k.yaml` and run `python train.py` and `python evaluate_ch.py`

- For CSL dataset and BERT+Decoder model, change the yaml path in `config.py` to `yamls/csl10k_with_pretrain.yaml` and run `python train_with_pretrained_encoder.py` and `python evaluate_ch_with_pretrained_model.py`

warn: some pretrained path in config need to be change
### preprocessed data
The preprocessed CSL data can be found in 链接：https://pan.baidu.com/s/1KEw25IJuj8ZLJD9JfjqzlQ 
提取码：3n5g</br>
The preprocessed multi30k data can be found in 链接：https://pan.baidu.com/s/1pgl_YQ3N6qv0s_pTPoAg2A 
提取码：k9wo

### pretrained model GAU
The pretrained model GAU_alpha can be download from [GAU_alpha](https://github.com/ZhuiyiTechnology/GAU-alpha)

Contact Info: panshuai2019ps@163.com