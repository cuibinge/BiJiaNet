import keras
from keras.layers import *
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dropout, Dense, RepeatVector, Lambda, Reshape, Conv3D, Conv2D, Flatten, InputSpec
from keras import layers
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    CompatibleExtractPatches,
    conv2d_no_bias,
    drop_block,
    layer_norm,
    mlp_block,
    add_pre_post_process,
)
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    ChannelAffine,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    #MixupToken,
    mlp_block,
    add_pre_post_process,
)
import output
import ptflops
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(128, 128),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.




def FExtractor(inputs):
    conv3d_shape = inputs.shape
    x = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3], 1))(inputs)

    x = Conv3D(filters=16, kernel_size=(1, 1, 3), activation='relu', padding='same')(x)
    x = Conv3D(filters=32, kernel_size=(1, 1, 5), activation='relu', padding='same')(x)

    conv3d_shape = x.shape
    x = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4]))(x)

    ##### #################################    UNet

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
    # x = layers.BatchNormalization()(x)

    x1 = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same")(
        previous_block_activation
    )
    x = layers.add([x1, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

    x = layers.Activation("relu")(x1)
    x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(2)(x)

    # Project residual
    residual = layers.UpSampling2D(2)(previous_block_activation)
    residual = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(residual)
    x = layers.add([x, residual])  # Add back residual

    return x

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
LAYER_NORM_EPSILON = 1e-6

def stem(inputs, stem_width, use_conv_stem=False, drop_rate=0, activation="gelu"):
    if use_conv_stem:

        nn = conv2d_no_bias(inputs, stem_width // 2, kernel_size=3, strides=2, padding="same", use_bias=True)
        nn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(nn)
        nn = activation_by_name(nn, activation, name=name)
        nn = conv2d_no_bias(nn, stem_width, kernel_size=3, strides=2, padding="same", use_bias=True, )
        nn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(nn)

    else:

        nn = conv2d_no_bias(inputs, stem_width, 4, strides=4, padding="valid", use_bias=True)
        nn = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)(nn)

    nn = keras.layers.Dropout(drop_rate) if drop_rate > 0 else nn

    return nn


class MultiHeadRelativePositionalKernelBias(tf.keras.layers.Layer):
    def __init__(self, input_height=-1, is_heads_first=False, **kwargs):
        super().__init__(**kwargs)
        self.input_height, self.is_heads_first = input_height, is_heads_first

    def build(self, input_shape):

        blocks, num_heads = (input_shape[2], input_shape[1]) if self.is_heads_first else (input_shape[1], input_shape[2])
        size = int(tf.math.sqrt(float(input_shape[-1])))
        height = self.input_height if self.input_height > 0 else int(tf.math.sqrt(float(blocks)))
        width = blocks // height
        pos_size = 2 * size - 1
        initializer = tf.initializers.truncated_normal(stddev=0.02)
        self.pos_bias = self.add_weight(name="positional_embedding", shape=(num_heads, pos_size * pos_size), initializer=initializer, trainable=True)

        idx_hh, idx_ww = tf.range(0, size), tf.range(0, size)
        coords = tf.reshape(tf.expand_dims(idx_hh, -1) * pos_size + idx_ww, [-1])
        bias_hh = tf.concat([idx_hh[: size // 2], tf.repeat(idx_hh[size // 2], height - size + 1), idx_hh[size // 2 + 1 :]], axis=-1)
        bias_ww = tf.concat([idx_ww[: size // 2], tf.repeat(idx_ww[size // 2], width - size + 1), idx_ww[size // 2 + 1 :]], axis=-1)
        bias_hw = tf.expand_dims(bias_hh, -1) * pos_size + bias_ww
        bias_coords = tf.expand_dims(bias_hw, -1) + coords
        bias_coords = tf.reshape(bias_coords, [-1, size**2])[::-1]  # torch.flip(bias_coords, [0])

        bias_coords_shape = [bias_coords.shape[0]] + [1] * (len(input_shape) - 4) + [bias_coords.shape[1]]
        self.bias_coords = tf.reshape(bias_coords, bias_coords_shape)  # [height * width, 1 * n, size * size]
        if not self.is_heads_first:
            self.transpose_perm = [1, 0] + list(range(2, len(input_shape) - 1))  # transpose [num_heads, height * width] -> [height * width, num_heads]

    def call(self, inputs):
        if self.is_heads_first:
            return inputs + tf.gather(self.pos_bias, self.bias_coords, axis=-1)
        else:
            return inputs + tf.transpose(tf.gather(self.pos_bias, self.bias_coords, axis=-1), self.transpose_perm)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"input_height": self.input_height, "is_heads_first": self.is_heads_first})
        return base_config


def LWA(
        inputs, kernel_size=3, num_heads=2, key_dim=0, out_weight=True, qkv_bias=True, out_bias=True, attn_dropout=0,
        output_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / (float(key_dim) ** 0.5)
    out_shape = cc
    qkv_out = num_heads * key_dim

    should_pad_hh, should_pad_ww = max(0, kernel_size - hh), max(0, kernel_size - ww)
    if should_pad_hh or should_pad_ww:
        inputs = tf.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])
        _, hh, ww, cc = inputs.shape

    qkv = keras.layers.Dense(qkv_out * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    query, key_value = tf.split(qkv, [qkv_out, qkv_out * 2], axis=-1)  # Matching weights from PyTorch
    query = tf.expand_dims(tf.reshape(query, [-1, hh * ww, num_heads, key_dim]),
                           -2)  # [batch, hh * ww, num_heads, 1, key_dim]

    key_value = CompatibleExtractPatches(sizes=kernel_size, strides=1, padding="VALID", compressed=False)(key_value)
    padded = (kernel_size - 1) // 2

    key_value = tf.concat(
        [tf.repeat(key_value[:, :1], padded, axis=1), key_value, tf.repeat(key_value[:, -1:], padded, axis=1)], axis=1)
    key_value = tf.concat(
        [tf.repeat(key_value[:, :, :1], padded, axis=2), key_value, tf.repeat(key_value[:, :, -1:], padded, axis=2)],
        axis=2)

    key_value = tf.reshape(key_value, [-1, kernel_size * kernel_size, key_value.shape[-1]])
    key, value = tf.split(key_value, 2,
                          axis=-1)  # [batch * block_height * block_width, kernel_size * kernel_size, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]),
                       [0, 2, 3, 1])  # [batch * hh*ww, num_heads, key_dim, kernel_size * kernel_size]
    key = tf.reshape(key, [-1, hh * ww, num_heads, key_dim,
                           kernel_size * kernel_size])  # [batch, hh*ww, num_heads, key_dim, kernel_size * kernel_size]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 1, 3])
    value = tf.reshape(value, [-1, hh * ww, num_heads, kernel_size * kernel_size,
                               key_dim])  # [batch, hh*ww, num_heads, kernel_size * kernel_size, key_dim]

    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale
    attention_scores = MultiHeadRelativePositionalKernelBias(input_height=hh, name=name and name + "pos")(
        attention_scores)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(
        attention_scores) if attn_dropout > 0 else attention_scores

    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.reshape(attention_output, [-1, hh, ww, num_heads * key_dim])

    if should_pad_hh or should_pad_ww:
        attention_output = attention_output[:, : hh - should_pad_hh, : ww - should_pad_ww, :]

    if out_weight:
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(
            attention_output)

    attention_output = keras.layers.Dropout(output_dropout, name=name and name + "out_drop")(
        attention_output) if output_dropout > 0 else attention_output

    return attention_output



def block(inputs, out_channel, num_heads=0, attn_kernel_size=3, qkv_bias=True, mlp_ratio=4, mlp_drop_rate=0,
          attn_drop_rate=0, drop_rate=0, gamma=-1, activation="gelu"):
    is_conv = False if num_heads > 0 else True  # decide by if num_heads > 0

    input_channel = inputs.shape[-1]  # Same with out_channel

    pos_emb1 = depthwise_conv2d_no_bias(inputs, kernel_size=1, padding="SAME", use_bias=True)
    pos_emb2 = depthwise_conv2d_no_bias(inputs, kernel_size=3, padding="SAME", use_bias=True)
    pos_emb3 = depthwise_conv2d_no_bias(inputs, kernel_size=5, padding="SAME", use_bias=True)

    pos_out = keras.layers.Add()([inputs, pos_emb1, pos_emb2, pos_emb3])

    # print(f">>>> {is_conv = }, {num_heads = }")
    if is_conv:

        attn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(pos_out)
        attn = conv2d_no_bias(attn, out_channel, 1, use_bias=True)
        attn = depthwise_conv2d_no_bias(attn, kernel_size=5, padding="SAME", use_bias=True)
        attn = conv2d_no_bias(attn, out_channel, 1, use_bias=True)

    else:

        attn = layer_norm(inputs)
        attn = LWA(attn, attn_kernel_size, num_heads, attn_dropout=attn_drop_rate)
        attn = ChannelAffine(use_bias=False, weight_init_value=gamma)(attn) if gamma >= 0 else attn

    attn = drop_block(attn)
    attn_out = keras.layers.Add()([inputs, pos_out, attn])

    if is_conv:

        mlp = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(attn_out)

    else:

        mlp = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)(attn_out)

    mlp = mlp_block(mlp, int(out_channel * mlp_ratio), activation=activation)

    mlp = ChannelAffine(use_bias=False, weight_init_value=gamma)(mlp) if gamma >= 0 else mlp

    return keras.layers.Add()([inputs, attn_out, mlp])



def WetMapFormer(
        num_blocks=[3, 4],
        out_channels=[128, 256],
        head_dimension=64,
        use_conv_stem=False,
        block_types=["conv", "transform"],
        stem_width=-1,
        qkv_bias=True,
        mlp_ratio=4,
        layer_scale=-1,
        mix_token=False,
        token_label_top=False,
        input_shape=(8, 8, 16, 1),
        num_classes=7,
        activation="gelu",
        mlp_drop_rate=0,
        attn_drop_rate=0,
        drop_connect_rate=0,
        dropout=0,
        classifier_activation="softmax",
        pretrained=None,
        model_name="WetMapFormer",
        kwargs=None,
):
    inputs1 = Input(input_shape)

    inputs = tf.transpose(inputs1, perm=[0, 2, 3, 1])

    augmented = data_augmentation(inputs)

    x = FExtractor(augmented)

    """ stem """
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = stem(x, stem_width, use_conv_stem, drop_rate=mlp_drop_rate,
              activation=activation)  # It's using mlp_drop_rate for stem

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        stack_name = "stack{}_".format(stack_id + 1)
        is_conv_block = True if block_type[0].lower() == "c" else False
        num_heads = 0 if is_conv_block else out_channel // head_dimension
        if stack_id > 0:
            if use_conv_stem:
                nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=2, padding="same", use_bias=True)
                nn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(nn)
            else:
                nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=2, use_bias=True)
                nn = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)(nn)

        for block_id in range(num_block):
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = block(nn, out_channel, num_heads, qkv_bias, mlp_ratio, attn_drop_rate, block_drop_rate, layer_scale,
                       activation)
            global_block_id += 1
    nn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(nn)

    """ output """
    if token_label_top and num_classes > 0:
        # Training with label token
        nn_cls = output_block(nn, num_classes=num_classes, classifier_activation=None)  # Don't use softmax here
        nn_aux = keras.layers.Dense(num_classes)(nn)

        if mix_token:

            nn_aux = mixup_token.do_mixup_token(nn_aux, bbox)
            nn_aux = keras.layers.Reshape((-1, nn_aux.shape[-1]), dtype="float32")(nn_aux)

            left, top, right, bottom = bbox
            lam = 1 - ((right - left) * (bottom - top) / (nn_aux.shape[1] * nn_aux.shape[2]))
            lam_repeat = tf.expand_dims(tf.repeat(lam, tf.shape(nn_cls)[0], axis=0), 1)
            nn_cls = keras.layers.Concatenate(axis=-1, dtype="float32")([nn_cls, lam_repeat])

        else:

            nn_aux = keras.layers.Reshape((-1, nn_aux.shape[-1]), dtype="float32")(nn_aux)

        out = [nn_cls, nn_aux]

    else:
        out = output.output_block(nn, num_classes=num_classes, classifier_activation=classifier_activation)
        # out = tf.transpose(out, perm=[0, 3, 1, 2])
    model = keras.models.Model(inputs1, out, name=model_name)

    return model



def WetMapFormer_small(input_shape=(8, 8, 16), num_classes=7, classifier_activation="softmax", token_label_top=False,
                       **kwargs):
    num_blocks = [1, 1]
    head_dimension = 32

    return WetMapFormer(**locals(), model_name="WetMapFormer_small", **kwargs)


import tensorflow as tf

def conv_flops(filters, kernel_size, input_shape, padding, strides=(1, 1)):
    # 假设 input_shape 是 (batch_size, height, width, channels)

    height, width, channels = input_shape
    # 根据 padding 和 strides 计算输出尺寸
    print(strides[0])
    print(padding[0])
    output_height = (height - kernel_size[0] + 2 * int(strides[0])) // int(strides[0]) + 1
    output_width = (width - kernel_size[1] + 2 * int(strides[0])) // int(strides[1]) + 1
    # 计算 FLOPs
    flops = (kernel_size[0] * kernel_size[1] * channels * filters + filters) * output_height * output_width
    return flops

def dense_flops(in_features, out_features):
    flops = (in_features * out_features + out_features)
    return flops

def calculate_flops(model):
    total_flops = 0
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D)):
            # 确保输入形状有效
            if layer.input_shape is None:
                # 这里可以添加错误处理或跳过该层
                continue
            # 对于 Conv3D，需要调整 input_shape 和 kernel_size 的处理
            if isinstance(layer, tf.keras.layers.Conv2D):
                input_shape = layer.input_shape[1:]  # 去掉 batch_size
                kernel_size = layer.kernel_size
                padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding,) * len(kernel_size)
                strides = layer.strides if isinstance(layer.strides, tuple) else (layer.strides,) * len(kernel_size)
                flops = conv_flops(layer.filters, kernel_size, input_shape, padding, strides)
            # 如果需要支持 Conv3D，请在这里添加相应的代码
        elif isinstance(layer, tf.keras.layers.Dense):
            in_features = layer.input_shape[-1]
            out_features = layer.units
            flops = dense_flops(in_features, out_features)
        # 对于其他类型的层，可以选择跳过或添加相应的 FLOPs 计算
        # ...
        if 'flops' in locals():  # 检查 flops 是否已经被赋值
            total_flops += flops

    return total_flops

# 注意：为了使用这个函数，您需要有一个已经构建好的模型，并且该模型应该具有有效的输入形状。
# 例如：
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
# print(calculate_flops(model))


# 构建模型（注意：这里只是示例，实际的模型构建需要根据给定的代码进行）
# model = WetMapFormer(...)
# 注意：由于 WetMapFormer 是一个复杂的模型，且包含多个自定义层和复杂的结构
# 你需要确保在调用 calculate_flops 函数之前，模型已经被正确构建并加载了输入数据
# 这样你才能准确地获取每一层的 input_shape 和其他必要的参数来计算 FLOPs

# 输出总 FLOPs





if __name__=='__main__':
    x = K.random_normal((16,4,128,128))
    model = WetMapFormer_small(input_shape=(4,128,128), num_classes=10)
    print("Total FLOPs:", calculate_flops(model))
    # flops, params = ptflops.get_model_complexity_info(model, (4, 128, 128), as_strings=True,
    #                                                   print_per_layer_stat=True, verbose=True)
    # print('FLOPs:  ' + flops)
    # print('Params: ' + params)
    # out = model(x)
    # print(out.shape)