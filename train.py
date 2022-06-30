from models import *
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import DataGenerator, sequence_padding, AutoRegressiveDecoder
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from config import config

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
        batch_source_ids, batch_target_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            source, target = d
            source_tokens = ['[CLS]'] + source.split() + ['[SEP]']
            source_token_ids = en_tokenizer.tokens_to_ids(source_tokens)
            target_tokens = ['[CLS]'] + target.split() + ['[SEP]']
            target_token_ids = de_tokenizer.tokens_to_ids(target_tokens)

            batch_source_ids.append(source_token_ids)
            batch_target_ids.append(target_token_ids[:-1])
            batch_labels.append(target_token_ids[1:])
            if len(batch_labels) == self.batch_size or is_end:
                batch_source_ids = sequence_padding(batch_source_ids)
                batch_target_ids = sequence_padding(batch_target_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_source_ids, batch_target_ids, batch_labels], None
                batch_source_ids, batch_target_ids, batch_labels = [], [], []


encode_layer = EncodeLayer(
    vocab_size=en_tokenizer._vocab_size,
    hidden_size=config.hidden_size,
    num_hidden_layers=config.n_layer,
    num_attention_heads=config.n_head,
    intermediate_size=config.inter_hidden_size,
    hidden_act='relu',
    dropout_rate=config.dropout_rate,
    attention_dropout_rate=config.attention_dropout_rate,
)
encode_layer.build()
if config.share_embedding:
    embedding = encode_layer.layers['Embedding-Token']
else:
    embedding = None
model = DecodeLayer
model = extend_with_language_model(model)
decode_layer = model(
    vocab_size=de_tokenizer._vocab_size,
    hidden_size=config.hidden_size,
    num_hidden_layers=config.n_layer,
    num_attention_heads=config.n_head,
    intermediate_size=config.inter_hidden_size,
    hidden_act='relu',
    dropout_rate=config.dropout_rate,
    attention_dropout_rate=config.attention_dropout_rate,
    prefix='Decoder',
    embedding=embedding
)
decode_layer.build()
seq2seq_output = decode_layer.call([decode_layer.inputs[0], encode_layer.output])
loss_fn = keras.losses.CategoricalCrossentropy(reduction='none')
class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        seq2seq_loss = self.compute_seq2seq_loss(inputs, mask)
        return seq2seq_loss

    def compute_seq2seq_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.greater(y_true, 0), K.floatx())
        one_hot = y_true
        if config.smooth:
            n_class = de_tokenizer._vocab_size
            one_hot = tf.one_hot(y_true, depth=n_class)
            one_hot = one_hot * (1 - 0.1) + (1 - one_hot) * 0.1 / (n_class - 1)
        loss = loss_fn(one_hot, y_pred)
        loss = tf.reduce_sum(loss*y_mask) / tf.reduce_sum(y_mask)
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = tf.reduce_sum(accuracy * y_mask) / tf.reduce_sum(y_mask)
        self.add_metric(accuracy, 'accuracy')

        return loss


label_input = Input(shape=(None,), dtype=tf.int32)
seq2seq_output = CrossEntropy([1])([label_input, seq2seq_output])
seq2seq_model = keras.models.Model([encode_layer.input, decode_layer.inputs[0], label_input], seq2seq_output)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, max_lr=None):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = step
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        if self.max_lr is not None:
            return tf.math.minimum(self.max_lr, lr)
        return lr
learning_rate = CustomSchedule(512, warmup_steps=4000)
seq2seq_model.compile(optimizer=Adam(learning_rate, beta_1=0.9, beta_2=0.998, epsilon=1e-9))
seq2seq_model.summary()

class AutoSeq(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        label_ids = output_ids
        return self.last_token(seq2seq_model).predict([token_ids, output_ids, label_ids])

    def generate(self, text, topk=1):
        text = ['[CLS]'] + text.split() + ['[SEP]']
        token_ids = np.array([en_tokenizer.tokens_to_ids(text)])
        output_ids = self.beam_search(token_ids,
                                      topk=topk)  # 基于beam search
        return de_tokenizer.ids_to_tokens(output_ids)

autoseq = AutoSeq(start_id=de_tokenizer._token_start_id, end_id=de_tokenizer._token_end_id, maxlen=config.max_len)

class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self):
        self.best_metric = 0.0

    def on_epoch_end(self, epoch, logs=None):
        print(f'true {target_text}')
        print('pred {}'.format(' '.join(autoseq.generate(text))))


if __name__ == '__main__':
    print(config.__dict__)
    en_data = open(config.train_source_dir, 'r', encoding='utf-8').read().splitlines()
    de_data = open(config.train_target_dir, 'r', encoding='utf-8').read().splitlines()
    data = list(zip(en_data, de_data))
    val_data = list(zip(
        open(config.val_source_dir, 'r', encoding='utf-8').read().splitlines(),
        open(config.val_target_dir, 'r', encoding='utf-8').read().splitlines()
    ))
    max_en_len = max([len(x.split()) for x in en_data])
    max_de_len = max([len(x.split()) for x in de_data])
    print(f'en maxlen {max_en_len}')
    print(f'de maxlen {max_de_len}')
    print(f'total trian numbers {len(data)}')

    train_loader = data_generator(data, batch_size=config.batch_size)
    dev_loader = data_generator(val_data, batch_size=config.batch_size)
    text = 'two young , white males are outside near many bushes .'
    target_text = 'zwei junge weiße männer sind im freien in der nähe vieler büsche .'
    evaluator = Evaluator()
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(config.save_model_path, save_weights_only=True,
                                                                   monitor='val_accuracy', save_best_only=True)

    # seq2seq_model.load_weights('saved_model_lc')
    seq2seq_model.fit(train_loader.forfit(), epochs=config.epochs, steps_per_epoch=train_loader.steps,
                      validation_data=dev_loader.forfit(), validation_steps=dev_loader.steps,
                      callbacks=[model_checkpoint_callback])
else:
    seq2seq_model.load_weights(config.save_model_path)
