from config import config
import os
if config.multi_card:
    os.environ['TF_KERAS'] = '1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder, sequence_padding
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm
from models import *
from pathlib import Path
from datetime import datetime
from asr_utils.features.speech_featurizers import NumpySpeechFeaturizer
from asr_utils.features.specaugment_numpy import Augmentation
from jiwer import wer

import tensorflow as tf
import numpy as np
import random
import pandas as pd
import soundfile as sf
import wavio
np.random.seed(config.seed)
tf.set_random_seed(config.seed)
random.seed(config.seed)
if config.multi_card:
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


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
featurizer = NumpySpeechFeaturizer({"preemphasis": 0.97})
augmentation = Augmentation({"time_masking": {"num_masks": 10,
                                                  "mask_factor": 100,
                                                  "p_upperbound": 0.05},
                                 "freq_masking": {"num_masks": 1,
                                                  "mask_factor": 27},
                                 'prob': 0.5})


def read_wav(filename):
    if '.wav' in filename:
        read_obj = wavio.read(filename)
        data = read_obj.data
        rate = read_obj.rate
    elif '.flac' in filename:
        data, rate = sf.read(filename, dtype=np.int16)
    assert rate == 16000
    return data, rate


class data_generator(DataGenerator):
    def __init__(self, run_mode=None, **kwargs):
        super(data_generator, self).__init__(**kwargs)
        self.run_mode = run_mode

    def __iter__(self, random=False):
        batch_feature, batch_label = [], []
        for is_end, d in self.sample(random):
            filename, target = d
            audio, rate = read_wav(filename)
            feature = featurizer.extract(audio.astype(np.float32))
            if self.run_mode == 'train':
                feature = augmentation.augment(feature)
            feature = np.squeeze(feature).tolist()
            target_token_ids, _ = de_tokenizer.encode(target, maxlen=config.max_t_len)

            if config.multi_card:
                yield feature, target_token_ids
            else:
                batch_feature.append(feature)
                batch_label.append(target_token_ids)
                if len(batch_feature) == self.batch_size or is_end:
                    batch_feature = sequence_padding(batch_feature)
                    batch_label = sequence_padding(batch_label)
                    yield [batch_feature, batch_label], None
                    batch_feature, batch_label = [], []

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
        if config.multi_card:
            self.add_metric(accuracy, name='acc', aggregation='mean')
        else:
            self.add_metric(accuracy, 'acc')

        return loss

def build_model():
    # 词表不同，将Encoder和Decoder分开创建
    encode_layer = ASRRoformerEncoder(
        vocab_size=None,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.enc_n_layer,
        num_attention_heads=config.n_head,
        intermediate_size=config.inter_hidden_size,
        hidden_act=config.hidden_act,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        max_position=512
    )
    decode_layer = _MODEL_NAME[config.model_name][1](
        vocab_size=de_tokenizer._vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.dec_n_layer,
        num_attention_heads=config.n_head,
        intermediate_size=config.inter_hidden_size,
        hidden_act=config.hidden_act,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        prefix='Decoder',  # 让编码层和解码层的名字不一样
        max_position=512
    )
    encode_layer.build()
    decode_layer.build()

    decoder_output = decode_layer.call(encode_layer.outputs + decode_layer.inputs[1:])
    seq2seq_output = CrossEntropy([1])(decode_layer.inputs[1:] + [decoder_output])
    seq2seq_model = keras.models.Model(encode_layer.inputs + decode_layer.inputs[1:], seq2seq_output)

    # 线性递增递减学习率
    Adamp = extend_with_piecewise_linear_lr(Adam)
    seq2seq_model.compile(optimizer=Adamp(learning_rate=config.learning_rate,
                                          lr_schedule={0: 0.0, config.warmup:1., 400000: 0.1},
                                          ))
    seq2seq_model.summary()
    return seq2seq_model


strategy = tf.distribute.MirroredStrategy()
if config.multi_card:
    with strategy.scope():
        seq2seq_model = build_model()
else:
    seq2seq_model = build_model()

class AutoSeq(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        feature = inputs[0]
        return self.last_token(seq2seq_model).predict([feature, output_ids])

    def generate(self, filename, topk=1):
        audio, rate = sf.read(filename, dtype=np.int16)
        feature = featurizer.extract(audio.astype(np.float32))
        feature = np.expand_dims(np.squeeze(feature), 0)
        output_ids = self.beam_search(feature,
                                      topk=topk)  # 基于beam search
        return de_tokenizer.decode(output_ids)

autoseq = AutoSeq(start_id=de_tokenizer._token_start_id, end_id=de_tokenizer._token_end_id, maxlen=config.max_t_len)

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
        rouge_1, rouge_2, rouge_l, bleu, cer = 0, 0, 0, 0, 0
        predictions = []
        targets = []
        for filename, target in tqdm(data):
            total += 1
            pred_target = autoseq.generate(filename, topk=1)
            target = target.lower()
            pred_target = ' '.join(pred_target).lower()
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
                cer += wer(target, pred_target)
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
        cer /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
            'cer': cer
        }


def load_data(filename):
    """加载数据，一行一条文本"""
    maxlen = 0
    max_filesize = 0
    out = []
    df = pd.read_csv(filename, header=0)
    for i in range(df.shape[0]):
        wav_filename = df.loc[i, 'wav_filename']
        wav_filesize = df.loc[i, 'wav_filesize']
        transcript = df.loc[i, 'transcript']
        maxlen = max(maxlen, len(transcript.split()))
        max_filesize = max(max_filesize, wav_filesize)
        if wav_filesize > 640044:  # 去掉大于20s的
            continue
        out.append((wav_filename, transcript))
    print(f"maxlen of {filename} is {maxlen}, max_filesize of {filename} is {max_filesize}, number of samples is {len(out)}")
    return out

if __name__ == '__main__':
    print(config.__dict__)
    if not os.path.exists(Path(config.save_model_path).parent):
        os.mkdir(Path(config.save_model_path).parent)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    data = load_data(config.train_dir)
    valid_data = load_data(config.val_dir)

    train_loader = data_generator(data=data, batch_size=config.batch_size, run_mode='train')  # 只有训练集需要加mode=train，加入数据增强
    valid_loader = data_generator(data=valid_data, batch_size=config.batch_size)
    evaluator = Evaluator(num_samples=config.eval_num_samples)
    if config.continue_train:
        seq2seq_model.load_weights(config.save_model_path)
    csv_logger = keras.callbacks.CSVLogger(f"{config.log_dir}train_{config.task_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(config.save_model_path,
                                                                save_weights_only=True,
                                                                monitor='val_acc', save_best_only=True)
    # 设置两种callback，一种是belu最大保存模型，比较慢，一种是直接按照准确率保存模型，比较快
    if config.train_mode:
        if not config.multi_card:
            if config.belu_callback:
                seq2seq_model.fit(train_loader.forfit(), epochs=config.epochs, steps_per_epoch=len(train_loader),
                                  callbacks=[evaluator, csv_logger], initial_epoch=config.initial_epoch)
            else:
                seq2seq_model.fit(train_loader.forfit(), epochs=config.epochs, steps_per_epoch=len(train_loader),
                                  validation_data=valid_loader.forfit(), validation_steps=len(valid_loader),
                                  callbacks=[model_checkpoint_callback, csv_logger], initial_epoch=config.initial_epoch)
        else:
            train_dataset = train_loader.to_dataset(
                types=('float32', 'int32'),
                shapes=([None, 80], [None]),
                names=('Encoder-Input-Feature', 'DecoderDecoder-Input-Token'),
                padded_batch=True
            )
            valid_dataset = valid_loader.to_dataset(
                types=('float32', 'int32'),
                shapes=([None, 80], [None]),
                names=('Encoder-Input-Feature', 'DecoderDecoder-Input-Token'),
                padded_batch=True
            )
            seq2seq_model.fit(train_dataset, epochs=config.epochs, steps_per_epoch=len(train_loader),
                              validation_data=valid_dataset, validation_steps=len(valid_loader),
                              callbacks=[model_checkpoint_callback, csv_logger])
    else:
        test_data = load_data(config.test_dir)
        seq2seq_model.load_weights(config.save_model_path)
        print(evaluator.evaluate(test_data))
else:
    seq2seq_model.load_weights(config.save_model_path)
