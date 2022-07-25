from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import DataGenerator, sequence_padding, AutoRegressiveDecoder
from bert4keras.optimizers import Adam, extend_with_parameter_wise_lr
from bert4keras.models import build_transformer_model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from config import config # use csl10k_with_pretrain.yaml
from rouge import Rouge
from tqdm import tqdm
from models import *
from pathlib import Path
from datetime import datetime

import os
import tensorflow as tf
import numpy as np
np.random.seed(config.seed)
tf.set_random_seed(config.seed)

# 中文文本生成，使用同一词表即可
tokenizer = Tokenizer(
    token_dict=config.token_dict,
    do_lower_case=True,
)

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_source_ids, batch_segment_ids, batch_target_ids = [], [], []
        for is_end, d in self.sample(random):
            source, target = d
            # 中文直接使用tokenizer.encode
            source_token_ids = tokenizer.encode(source, maxlen=config.max_c_len)[0] # [token_ids, segment_ids]
            target_token_ids = tokenizer.encode(target, maxlen=config.max_t_len)[0]
            segment_ids = [0] * len(source_token_ids)

            batch_source_ids.append(source_token_ids)
            batch_target_ids.append(target_token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_source_ids) == self.batch_size or is_end:
                batch_source_ids = sequence_padding(batch_source_ids)
                batch_target_ids = sequence_padding(batch_target_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_source_ids, batch_segment_ids, batch_target_ids], None
                batch_source_ids, batch_segment_ids, batch_target_ids = [], [], []


encoder = build_transformer_model(
    config_path=config.config_path,
    checkpoint_path=config.checkpoint_path,
    model=GAU_alpha
)
decoder = DecodeLayer(
    vocab_size=tokenizer._vocab_size,
    hidden_size=config.hidden_size,
    num_hidden_layers=config.dec_n_layer,
    num_attention_heads=config.n_head,
    intermediate_size=config.inter_hidden_size,
    hidden_act='gelu',
    dropout_rate=config.dropout_rate,
    attention_dropout_rate=config.attention_dropout_rate,
    prefix='Decoder',  # 让编码层和解码层的名字不一样
)
decoder.build()
loss_fn = keras.losses.CategoricalCrossentropy(reduction='none')
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
            n_class = tokenizer._vocab_size
            y_true = K.cast(y_true, tf.int32)
            one_hot = tf.one_hot(y_true, depth=n_class)
            one_hot = one_hot * (1 - 0.1) + (1 - one_hot) * 0.1 / (n_class - 1)
        loss = K.sparse_categorical_crossentropy(one_hot, y_pred)
        loss = tf.reduce_sum(loss*y_mask) / tf.reduce_sum(y_mask)
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = tf.reduce_sum(accuracy * y_mask) / tf.reduce_sum(y_mask)
        self.add_metric(accuracy, 'acc')

        return loss

outputs = decoder.call(encoder.outputs + decoder.inputs[1:])
seq2seq_output = CrossEntropy([1])(decoder.inputs[1:]+[outputs])
seq2seq_model = keras.models.Model(encoder.inputs+decoder.inputs[1:], seq2seq_output)

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
        token_ids = np.array([tokenizer.encode(text, maxlen=config.max_c_len)[0]])
        output_ids = self.beam_search(token_ids,
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)

autoseq = AutoSeq(start_id=tokenizer._token_start_id, end_id=tokenizer._token_end_id, maxlen=config.max_t_len)

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
            target = ' '.join(target).lower()
            pred_target = ' '.join(autoseq.generate(source, topk=1)).lower()
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
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            title, content = l.strip().split('\t')
            D.append((content, title))
    return D

if __name__ == '__main__':
    print(config.__dict__)
    if not os.path.exists(Path(config.save_model_path).parent):
        os.mkdir(Path(config.save_model_path).parent)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    data = load_data(config.train_dir)
    valid_data = load_data(config.val_dir)

    train_loader = data_generator(data, batch_size=config.batch_size)
    evaluator = Evaluator(num_samples=config.eval_num_samples)
    if config.continue_train:
        seq2seq_model.load_weights(config.save_model_path)
    csv_logger = keras.callbacks.CSVLogger(f"{config.log_dir}training_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
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
        test_data = load_data(config.test_dir)
        seq2seq_model.load_weights(config.save_model_path)
        print(evaluator.evaluate(test_data))
else:
    seq2seq_model.load_weights(config.save_model_path)
