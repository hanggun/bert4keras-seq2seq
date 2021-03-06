# bert4keras-seq2seq
An implementation of Seq2Seq model using bert4keras

This repository gives applications of Seq2Seq for multiple datasets, including multi30k-en-de (machine translation dataset), CSL (title generation dataset) and Aishell1, Primewords and other chinese Speech data (Automatic Speech Recognition Task)

For multi30k-en-de dataset, we use BPE tokenization, English dataset and German dataset do not share the same embedding. For CSL chinese dataset and ASR dataset, we use tokenizer with GAU vocab.

- For machine translation dataset, we test the transformer, transformer+ and transformer++ architecture; 
- For CSL NLG dataset, in addition to transformer, we also apply a BERT+Decoder architecture. *Note* that BERT+Decoder can achieve a similar performance with Unilm version. The reason that Seq2Seq model does not perform well in CSL dataset is that the training datase is small and in autoregressive mode, previous wrong word will have great impact on the subsequent predicted words.
- For ASR dataset, We test the transformer++ architecture, the prediction speed is 2 iter/s which is fast and the result is only cover 100 samples. For more samples, the CER can under 0.1.

The result can be seen below, The *Transformer+* is transformer+RoPE, The *Transformer++* is transformer+RoPE+GLU, The symbol *w/o smooth* is without smooth in crossentropy loss.:</br>
|model|dataset|rouge-1|rouge-2|rouge-l|bleu|CER|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|seq2seq(Transformer)|multi30k-en-de-test2016|0.6817|0.4727|0.6551|0.3547|-|
|seq2seq(Transformer+)|multi30k-en-de-test2016|0.6898|**0.4850**|0.6628|**0.3679**|-|
|seq2seq(Transformer+ w/o smooth)|multi30k-en-de-test2016|0.6790|0.4688|0.6505|0.3527|-|
|seq2seq(Transformer++)|multi30k-en-de-test2016|0.6895|0.4835|0.6640|0.3649|-|
|BERT+Roformer|multi30k-en-de-test2016|**0.6908**|0.4772|**0.6644**|0.3589|-|
|seq2seq|CSL-test|0.4366|0.2528|0.3853|0.1517|-|
|BERT+Deocder|CSL-test|0.6597|0.5384|0.6272|0.4386|-|
|seq2seq(Transformer++)|Aishell1-test|0.9279|0.8834|0.9279|0.8456|0.1131|

### Some Conclusion
1. BERT encoder with Roformer Decoder has the best R-1 and R-l, demonstrate the strong representative ability of pretrained model
2. Despite *Transformer++* does not outperform *Transformer+*, they have similar similar performance and in some other cases, I believe *Transformer++* is better than *Transformer+*
3. *Transformer+* give a significant boost of 1.32 point in the BLEU compared to Transformer
4. smooth play an important role in machine translation, without which the point of bleu decrease 1.52

### Environments
- bert4keras==0.11.3
- tensorflow-gpu==1.15.5 (We use tf1 backend and pure keras because it is much faster than tf.keras)
### Updates 2022.7.22
- To simplify the training and testing procedure, the train file is reconstructed. Now, just set `is_train` mode in config file, choose the right `.yaml` file, then run `python seq2seq_for_multi30k.py` or `python seq2seq_for_csl.py` or `python seq2seq_of_gau_encoder_for_csl` or `seq2seq_of_bert_encoder_for_multi30k.py` and you can get the similar result for machine translation task and NLG task.
### preprocessed data
The preprocessed CSL data can be found in ?????????https://pan.baidu.com/s/1KEw25IJuj8ZLJD9JfjqzlQ 
????????????3n5g</br>
The preprocessed multi30k data can be found in ?????????https://pan.baidu.com/s/1pgl_YQ3N6qv0s_pTPoAg2A 
????????????k9wo

### pretrained model GAU
The pretrained model GAU_alpha can be download from [GAU_alpha](https://github.com/ZhuiyiTechnology/GAU-alpha)

### Use bert4keras-seq2seq in V0.0.1
- just run `python train.py` and `python evaluate_bpe_data.py` and you can train and evaluate a multi30k machine translation model. All the configuration is written in `yamls/multi30k.yaml`.

- For CSL dataset and seq2seq model, change the yaml path in `config.py` to `yamls/csl10k.yaml` and run `python train.py` and `python evaluate_ch.py`

- For CSL dataset and BERT+Decoder model, change the yaml path in `config.py` to `yamls/csl10k_with_pretrain.yaml` and run `python train_with_pretrained_encoder.py` and `python evaluate_ch_with_pretrained_model.py`

warn: some pretrained path in config need to be change

Contact Info: panshuai2019ps@163.com