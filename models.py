#! -*- coding: utf-8 -*-
# 基于bert4keras的seq2seq模型
from bert4keras.layers import *
from bert4keras.models import Transformer, extend_with_language_model, RoFormerV2


class EncodeLayer(Transformer):
    """
    从Transformer基类继承
    """
    def __init__(self, **kwargs):
        super(EncodeLayer, self).__init__(**kwargs)  # 将多余参数送到父类进行初始化

    def get_inputs(self):
        """重写get_inputs，输入仅为token_ids"""
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
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
            function=lambda x: x*self.hidden_size**0.5,
            name='Encode-Scale'
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
            name='Embedding-Norm'
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )

        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """重写apply_main_layers，主要为multihead-attention，add and norm， FFN，add and norm"""
        x = inputs
        z = self.layer_norm_conds[0]
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index

        xi, x, arguments = x, [x, x, x], {'a_bias': None}

        # 普通的多层注意力机制
        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
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


class DecodeLayer(Transformer):
    """
    从Transformer基类继承
    """
    def __init__(self,
                 with_mlm=False,
                 embedding=None,  # 是否选择跳过最终层，跳过可以使用编码器的嵌入层进行输出
                 **kwargs):
        super(DecodeLayer, self).__init__(**kwargs)  # 将多余参数送到父类进行初始化
        self.with_mlm = with_mlm
        self.embedding = embedding

    def get_inputs(self):
        """重写get_inputs，输入为target token_ids和encoder output"""
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        encoder_output = self.apply(
            layer=Input, shape=(self.sequence_length, self.hidden_size), name='Encoder-Output'
        )
        inputs = [x_in, encoder_output]

        return inputs

    def apply_embeddings(self, inputs):
        """重写apply_embeddings，为target embedding加入position bias，encoder-output直接输出"""
        inputs = inputs[:]  # 浅拷贝
        x = inputs.pop(0)
        y = inputs.pop(0)
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
            name='Encode-Scale'
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
            name='Embedding-Norm'
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )

        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return [x, y]

    def apply_main_layers(self, inputs, index):
        """重写apply_main_layers，主要为masked-multihead-attention->add and norm->multihead-attention->add and norm
        ->FFN->add and norm"""
        inputs = inputs[:]  # 浅拷贝
        x = inputs.pop(0)
        encoder_output = inputs.pop(0)
        z = self.layer_norm_conds[0]
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index

        # 第一层self-attention的mask为时序mask，避免预测时看到后面的信息
        attention_mask = self.compute_attention_bias(index)
        xi, x, arguments = x, [x, x, x], {'a_bias': True}
        x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
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
        xi, x, arguments = x, [x, encoder_output, encoder_output], {'a_bias': None}

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
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

        return [x, encoder_output]

    def apply_final_layers(self, inputs):
        """直接输出"""
        inputs = inputs[:]
        x = inputs.pop(0)

        if self.embedding:
            x = self.embedding(x, mode='dense')
        else:
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
    model = DecodeLayer
    model = extend_with_language_model(model)
    decode_layer = model(
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
    inputs = np.ones((20,256))
    target_inputs = np.ones((20, 250))
    print(np.array(seq2seq_model.predict([inputs, target_inputs])))
