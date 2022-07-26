from config import config
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator, sequence_padding, AutoRegressiveDecoder, K, keras
from bert4keras.optimizers import Adam, extend_with_parameter_wise_lr
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm
from models import *
from pathlib import Path
from datetime import datetime

import os
import tensorflow as tf
import numpy as np
import random
np.random.seed(config.seed)
tf.set_random_seed(config.seed)
random.seed(config.seed)

_MODEL_NAME = {
    'transformer': [EncodeLayer, DecodeLayer],
    'roformer': [RoformerEncoder, RoformerDecoder]
}
# 如果使用相同的词表，则使用相同的token_dict即可
en_tokenizer = Tokenizer(
    token_dict=config.en_token_dict,
    do_lower_case=True,
)
de_tokenizer = Tokenizer(
    token_dict=config.de_token_dict,
    do_lower_case=True
)

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_source_ids, batch_segment_ids, batch_target_ids = [], [], []
        for is_end, d in self.sample(random):
            source, target = d
            # 英文BPE存在@，所以不能直接encode
            source_token_ids, segment_ids = en_tokenizer.encode(source, maxlen=config.max_c_len)
            target_tokens = ['[CLS]'] + target.strip().split()[:config.max_t_len-2] + ['[SEP]']
            target_token_ids = de_tokenizer.tokens_to_ids(target_tokens)

            batch_source_ids.append(source_token_ids)
            batch_target_ids.append(target_token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_source_ids) == self.batch_size or is_end:
                batch_source_ids = sequence_padding(batch_source_ids)
                batch_target_ids = sequence_padding(batch_target_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_source_ids, batch_segment_ids, batch_target_ids], None
                batch_source_ids, batch_segment_ids, batch_target_ids = [], [], []

# 词表不同，将Encoder和Decoder分开创建
encode_layer = build_transformer_model(
    config_path=config.config_path,
    checkpoint_path=config.checkpoint_path
)
decode_layer = _MODEL_NAME[config.model_name][1](
    vocab_size=de_tokenizer._vocab_size,
    hidden_size=config.hidden_size,
    num_hidden_layers=config.n_layer,
    num_attention_heads=config.n_head,
    intermediate_size=config.inter_hidden_size,
    hidden_act=config.hidden_act,
    dropout_rate=config.dropout_rate,
    attention_dropout_rate=config.attention_dropout_rate,
    prefix='Decoder',  # 让编码层和解码层的名字不一样
    max_position=512
)
decode_layer.build()
if config.smooth:
    loss_fn = keras.losses.CategoricalCrossentropy(reduction='none')
else:
    loss_fn = K.sparse_categorical_crossentropy
class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_true = y_true[:, 1:]
        y_pred = y_pred[:, :-1] # 预测从第一位开始
        y_mask = K.cast(K.greater(y_true, 0), K.floatx())
        one_hot = y_true
        if config.smooth:
            n_class = de_tokenizer._vocab_size
            y_true = K.cast(y_true, tf.int32)
            one_hot = tf.one_hot(y_true, depth=n_class)
            one_hot = one_hot * (1 - 0.1) + (1 - one_hot) * 0.1 / (n_class - 1)
        loss = loss_fn(one_hot, y_pred)
        loss = tf.reduce_sum(loss*y_mask) / tf.reduce_sum(y_mask)
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = tf.reduce_sum(accuracy * y_mask) / tf.reduce_sum(y_mask)
        self.add_metric(accuracy, 'acc')

        return loss

decoder_output = decode_layer.call(encode_layer.outputs + decode_layer.inputs[1:])
seq2seq_output = CrossEntropy([1])(decode_layer.inputs[1:] + [decoder_output])
seq2seq_model = keras.models.Model(encode_layer.inputs + decode_layer.inputs[1:], seq2seq_output)

# bert的学习率为3e-5，decoder的学习率为1e-4
Adamp = extend_with_parameter_wise_lr(Adam)
seq2seq_model.compile(optimizer=Adamp(learning_rate=config.learning_rate,
                                      paramwise_lr_schedule={'Decoder': 10/3},
                                      beta_1=0.9,
                                      beta_2=0.998,
                                      epsilon=1e-9))
seq2seq_model.summary()

class AutoSeq(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        segment_ids = np.array([[0] * token_ids.shape[1]])
        return self.last_token(seq2seq_model).predict([token_ids, segment_ids, output_ids])

    def generate(self, text, topk=1):
        token_ids = np.array([en_tokenizer.encode(text, maxlen=config.max_c_len)[0]])
        output_ids = self.beam_search(token_ids,
                                      topk=topk)  # 基于beam search
        return de_tokenizer.ids_to_tokens(output_ids)

autoseq = AutoSeq(start_id=de_tokenizer._token_start_id, end_id=de_tokenizer._token_end_id, maxlen=config.max_t_len)

def process_tokens(tokens):
    out_tokens = ''
    for token in tokens:
        if token[-2:] == '@@':
            out_tokens += token[:-2]
        else:
            out_tokens += token + ' '
    return out_tokens.split()

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    from https://github.com/ZhuiyiTechnology/t5-pegasus/blob/main/finetune.py
    """
    def __init__(self, num_samples=None):
        """
        :param num_samples: 决定多少样本进行验证
        """
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            seq2seq_model.save_weights(config.save_model_path)  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        if not self.num_samples:
            self.num_samples = len(data)+1
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        predictions = []
        targets = []
        for source, target in tqdm(data):
            total += 1
            pred_target = autoseq.generate(source, topk=1)[1:-1]
            target = ' '.join(process_tokens(target.split())).lower()
            pred_target = ' '.join(process_tokens(pred_target)).lower()
            if pred_target.strip():
                scores = self.rouge.get_scores(hyps=pred_target, refs=target)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[target.split(' ')],
                    hypothesis=pred_target.split(' '),
                    smoothing_function=self.smooth
                )
            targets.append(target)
            predictions.append(pred_target)
            if total == self.num_samples:
                break
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


def load_data(filename):
    """加载数据，一行一条文本"""
    out = []
    maxlen = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            out.append(line)
            maxlen = max(maxlen, len(line.split()))
    print(f"maxlen of {filename} is {maxlen}, number of samples is {len(out)}")
    return out

if __name__ == '__main__':
    print(config.__dict__)
    if not os.path.exists(Path(config.save_model_path).parent):
        os.mkdir(Path(config.save_model_path).parent)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    data = list(zip(
        load_data(config.train_source_dir),
        load_data(config.train_target_dir)
    ))
    valid_data = list(zip(
        load_data(config.val_source_dir),
        load_data(config.val_target_dir)
    ))

    train_loader = data_generator(data, batch_size=config.batch_size)
    evaluator = Evaluator(num_samples=config.eval_num_samples)
    if config.continue_train:
        seq2seq_model.load_weights(config.save_model_path)
    csv_logger = keras.callbacks.CSVLogger(f"{config.log_dir}train_{config.task_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    # 设置两种callback，一种是belu最大保存模型，比较慢，一种是直接按照准确率保存模型，比较快
    if config.train_mode:
        if config.belu_callback:
            seq2seq_model.fit(train_loader.forfit(), epochs=config.epochs, steps_per_epoch=len(train_loader),
                              callbacks=[evaluator, csv_logger], initial_epoch=config.initial_epoch)
        else:
            valid_loader = data_generator(valid_data, batch_size=config.batch_size)
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(config.save_model_path,
                                                                           save_weights_only=True,
                                                                           monitor='val_acc', save_best_only=True)
            seq2seq_model.fit(train_loader.forfit(), epochs=config.epochs, steps_per_epoch=len(train_loader),
                              validation_data=valid_loader.forfit(), validation_steps=len(valid_loader),
                              callbacks=[model_checkpoint_callback, csv_logger], initial_epoch=config.initial_epoch)
    else:
        test_data = list(zip(
            load_data(config.test_source_dir),
            load_data(config.test_target_dir)
        ))
        seq2seq_model.load_weights(config.save_model_path)
        print(evaluator.evaluate(test_data))
else:
    seq2seq_model.load_weights(config.save_model_path)
