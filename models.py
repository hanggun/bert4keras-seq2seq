#! -*- coding: utf-8 -*-
# 基于bert4keras的seq2seq模型
from bert4keras.layers import *
from bert4keras.models import Transformer, RoFormerV2, LM_Mask, Model


class EncodeLayer(Transformer):
    """
    从Transformer基类继承
    """

    def __init__(self, **kwargs):
        super(EncodeLayer, self).__init__(**kwargs)  # 将多余参数送到父类进行初始化

    def get_inputs(self):
        """重写get_inputs，输入仅为token_ids"""
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Encoder-Input-Token'
        )
        inputs = [x_in]

        return inputs

    def apply_embeddings(self, inputs):
        """重写apply_embeddings，为输入加入position bias"""
        inputs = inputs[:]  # 浅拷贝
        x = inputs.pop(0)
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=lambda x: x * self.hidden_size ** 0.5,
            name='Scale'
        )

        x = self.apply(
            inputs=x,
            layer=SinusoidalPositionEmbedding,
            output_dim=self.embedding_size,
            name='Embedding-Position'
        )

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Encoder-Embedding-Norm'
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Encoder-Embedding-Dropout'
        )

        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Encoder-Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """重写apply_main_layers，主要为multihead-attention，add and norm， FFN，add and norm"""
        x = inputs
        z = self.layer_norm_conds[0]
        attention_name = 'Encoder-Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Encoder-Transformer-%d-FeedForward' % index

        xi = x
        # self-attention
        x = self.apply(
            inputs=[x, x, x],
            layer=MultiHeadAttention,
            arguments={
                'a_bias': None
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """直接输出"""
        return inputs


class DecodeLayer(LM_Mask, Transformer):
    """
    从Transformer基类继承
    """

    def __init__(self,
                 **kwargs):
        super(DecodeLayer, self).__init__(**kwargs)  # 将多余参数送到父类进行初始化

    def get_inputs(self):
        """重写get_inputs，输入为target token_ids和encoder output"""
        x = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Decoder-Input-Token'
        )
        c = self.apply(
            layer=Input, shape=(self.sequence_length, self.hidden_size), name='Input-Context'
        )
        inputs = [c, x]

        return inputs

    def apply_embeddings(self, inputs):
        """重写apply_embeddings，为target embedding加入position bias，encoder-output直接输出"""
        c, x = inputs
        z = self.layer_norm_conds[0]

        # Embedding token, scale 和 sinusoidal 共用
        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=lambda x: x * self.hidden_size ** 0.5,
            name='Scale'
        )

        x = self.apply(
            inputs=x,
            layer=SinusoidalPositionEmbedding,
            output_dim=self.embedding_size,
            name='Embedding-Position'
        )

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Decoder-Embedding-Norm'
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Embedding-Dropout'
        )

        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Decoder-Embedding-Mapping'
            )

        return [c, x]

    def apply_main_layers(self, inputs, index):
        """重写apply_main_layers，主要为masked-multihead-attention->add and norm->multihead-attention->add and norm
        ->FFN->add and norm"""
        c, x = inputs
        z = self.layer_norm_conds[0]
        attention_name = 'Decoder-Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Decoder-Transformer-%d-FeedForward' % index

        # 第一层self-attention的mask为时序mask，避免预测时看到后面的信息
        attention_mask = self.compute_attention_bias(index)
        xi = x
        x = self.apply(
            inputs=[x, x, x, attention_mask],
            layer=MultiHeadAttention,
            arguments={
                'a_bias': True
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # 第二层cross-attention
        xi = x

        x = self.apply(
            inputs=[x, c, c],
            layer=MultiHeadAttention,
            arguments={
                'a_bias': None
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )
        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return [c, x]

    def apply_final_layers(self, inputs):
        """直接输出"""
        c, x = inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            arguments={'mode': 'dense'},
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x, layer=ScaleOffset, scale=False, name='Seq2seq-Bias'
        )
        x = self.apply(
            inputs=x,
            layer=Activation,
            activation='softmax',
            name='Seq2seq-Activation'
        )

        return x

    def compute_attention_bias(self, inputs=None):
        """修改LM Mask的序列长度（从 self.inputs[0] 改为 self.inputs[1] ）
        """
        old_inputs = self.inputs[:]
        self.inputs = [old_inputs[1]]
        mask = super(DecodeLayer, self).compute_attention_bias(inputs)
        self.inputs = old_inputs
        return mask


class Seq2Seq(Transformer):
    def __init__(self, **kwargs):
        super(Seq2Seq, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Encoder', 'Decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        del kwargs['num_hidden_layers']
        self._encoder = EncodeLayer(name=e_name,
                                    num_hidden_layers=kwargs['enc_num_hidden_layers'],
                                    **kwargs)
        self._decoder = DecodeLayer(name=d_name,
                                    num_hidden_layers=kwargs['dec_num_hidden_layers'],
                                    **kwargs)

    def build(self, **kwargs):
        """同时构建Encoder和Decoder
        """
        self._encoder.build(**kwargs)
        self._decoder.build(**kwargs)
        self.encoder = self._encoder.model
        self.decoder = self._decoder.model
        self.inputs = self.encoder.inputs + self.decoder.inputs[1:]
        self.outputs = self._decoder.call(
            self.encoder.outputs + self.decoder.inputs[1:]
        )
        self.model = Model(self.inputs, self.outputs)


class GAU_alpha(RoFormerV2):
    """GAU-α
    改动：基本模块换成GAU
    链接：https://kexue.fm/archives/9052
    """

    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        return super(GAU_alpha, self).initializer(shape, dtype, order, gain)

    def apply_main_layers(self, inputs, index):
        """GAU-α 的主体是基于Gated Attention Unit的模块
        顺序：GAU  --> Add --> LN
        """
        x = inputs

        attention_name = 'Transformer-%d-GatedAttentionUnit' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % attention_name
        )

        return x

    def variable_mapping(self):
        """重新定义权重映射
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'GAU_alpha/encoder/layer_%d/' % i
            mapping['Transformer-%d-GatedAttentionUnit' % i] = [
                prefix + 'gau/i_dense/kernel',
                # prefix + 'gau/i_dense/bias',
                prefix + 'gau/o_dense/kernel',
                # prefix + 'gau/o_dense/bias',
                # prefix + 'gau/q_scaleoffset/beta',
                prefix + 'gau/q_scaleoffset/gamma',
                # prefix + 'gau/k_scaleoffset/beta',
                prefix + 'gau/k_scaleoffset/gamma',
            ]

        return mapping


if __name__ == '__main__':
    import os
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    encode_layer = EncodeLayer(
        vocab_size=1000,
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_act='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1
    )
    decode_layer = DecodeLayer(
        vocab_size=1200,
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_act='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        prefix='Decoder'
    )
    encode_layer.build()
    decode_layer.build()
    seq2seq_output = decode_layer.call([decode_layer.inputs[0], encode_layer.output])
    seq2seq_model = keras.models.Model([encode_layer.input, decode_layer.inputs[0]], seq2seq_output)
    # inter_model = keras.Model(model.inputs, encode_layer.layers['Transformer-0-MultiHeadSelfAttention'].output)
    inputs = np.ones((20, 256))
    target_inputs = np.ones((20, 250))
    print(np.array(seq2seq_model.predict([inputs, target_inputs])))
