"""Musify models trained from scratch on fixed-size log-mel spectrograms."""

import tensorflow as tf

from .config import AudioConfig


def _conv_norm_swish(inputs, filters: int, kernel_size, *, strides=1, name: str):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.Activation("swish", name=f"{name}_swish")(x)


def _multiscale_residual_block(inputs, filters: int, *, strides=1, name: str):
    """Fuse short and wide time-frequency patterns with an independent residual block."""
    shortcut = tf.keras.layers.Conv2D(
        filters, 1, strides=strides, padding="same", use_bias=False, name=f"{name}_skip_conv"
    )(inputs)
    shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)

    branch_width = filters // 2
    short = tf.keras.layers.SeparableConv2D(
        branch_width,
        (3, 3),
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name=f"{name}_short_conv",
    )(inputs)
    short = tf.keras.layers.BatchNormalization(name=f"{name}_short_bn")(short)
    short = tf.keras.layers.Activation("swish", name=f"{name}_short_swish")(short)

    wide = tf.keras.layers.SeparableConv2D(
        branch_width,
        (3, 7),
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name=f"{name}_wide_conv",
    )(inputs)
    wide = tf.keras.layers.BatchNormalization(name=f"{name}_wide_bn")(wide)
    wide = tf.keras.layers.Activation("swish", name=f"{name}_wide_swish")(wide)

    x = tf.keras.layers.Concatenate(name=f"{name}_scale_fusion")([short, wide])
    x = tf.keras.layers.Conv2D(
        filters, 1, padding="same", use_bias=False, kernel_initializer="he_normal",
        name=f"{name}_projection",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_projection_bn")(x)

    gate = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_channel_pool")(x)
    gate = tf.keras.layers.Dense(
        max(filters // 8, 8), activation="swish", name=f"{name}_channel_reduce"
    )(gate)
    gate = tf.keras.layers.Dense(
        filters, activation="sigmoid", name=f"{name}_channel_gate"
    )(gate)
    gate = tf.keras.layers.Reshape((1, 1, filters), name=f"{name}_channel_reshape")(gate)
    x = tf.keras.layers.Multiply(name=f"{name}_channel_scale")([x, gate])
    x = tf.keras.layers.Add(name=f"{name}_residual_add")([shortcut, x])
    x = tf.keras.layers.Activation("swish", name=f"{name}_output")(x)
    return tf.keras.layers.SpatialDropout2D(0.04, name=f"{name}_spatial_dropout")(x)


def build_multiscale_model(config: AudioConfig, class_count: int) -> tf.keras.Model:
    """Create Musify's original multi-scale residual CNN; all weights start untrained."""
    inputs = tf.keras.Input(
        (config.n_mels, config.frames_per_clip, 1),
        name="normalized_log_mel",
    )
    x = _conv_norm_swish(inputs, 32, (5, 5), strides=(2, 2), name="stem")

    for stage, (filters, strides) in enumerate(
        ((48, 1), (96, 2), (160, 2), (224, 2)), start=1
    ):
        x = _multiscale_residual_block(
            x, filters, strides=strides, name=f"multiscale_stage_{stage}"
        )

    average = tf.keras.layers.GlobalAveragePooling2D(name="mean_time_frequency_pool")(x)
    maximum = tf.keras.layers.GlobalMaxPooling2D(name="max_time_frequency_pool")(x)
    x = tf.keras.layers.Concatenate(name="global_pool_fusion")([average, maximum])
    x = tf.keras.layers.Dense(
        192,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(5e-5),
        name="genre_projection",
    )(x)
    x = tf.keras.layers.Dropout(0.30, name="genre_dropout")(x)
    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", name="genre_probabilities"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="musify_multiscale_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_crnn_model(config: AudioConfig, class_count: int) -> tf.keras.Model:
    """Build the original convolutional recurrent baseline for comparison."""
    inputs = tf.keras.Input(
        (config.n_mels, config.frames_per_clip, 1),
        name="log_mel",
    )
    x = inputs
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    x = tf.keras.layers.Permute((2, 1, 3), name="time_first_features")(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten(), name="flatten_frequency_per_frame"
    )(x)
    x = tf.keras.layers.Dense(128, use_bias=False, name="frame_projection")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            96, return_sequences=True, dropout=0.2, recurrent_dropout=0.0
        ),
        name="bidirectional_gru",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="temporal_pooling")(x)
    x = tf.keras.layers.Dense(
        128, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="genre_projection"
    )(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", name="genre_probabilities"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="music_genre_crnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _crnn_residual_block(inputs, filters: int, *, strides=(1, 1), name: str):
    """Residual depthwise-separable block for a compact time-frequency encoder."""
    shortcut = tf.keras.layers.Conv2D(
        filters, 1, strides=strides, padding="same", use_bias=False,
        name=f"{name}_skip_conv",
    )(inputs)
    shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)

    x = tf.keras.layers.SeparableConv2D(
        filters, 3, strides=strides, padding="same", use_bias=False,
        depthwise_initializer="he_normal", pointwise_initializer="he_normal",
        name=f"{name}_sepconv_1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_1")(x)
    x = tf.keras.layers.Activation("swish", name=f"{name}_swish_1")(x)
    x = tf.keras.layers.SeparableConv2D(
        filters, 3, padding="same", use_bias=False,
        depthwise_initializer="he_normal", pointwise_initializer="he_normal",
        name=f"{name}_sepconv_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_2")(x)

    gate = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_se_pool")(x)
    gate = tf.keras.layers.Dense(
        max(filters // 8, 8), activation="swish", name=f"{name}_se_reduce"
    )(gate)
    gate = tf.keras.layers.Dense(filters, activation="sigmoid", name=f"{name}_se_gate")(gate)
    gate = tf.keras.layers.Reshape((1, 1, filters), name=f"{name}_se_reshape")(gate)
    x = tf.keras.layers.Multiply(name=f"{name}_se_scale")([x, gate])
    x = tf.keras.layers.Add(name=f"{name}_residual_add")([shortcut, x])
    x = tf.keras.layers.Activation("swish", name=f"{name}_output")(x)
    return tf.keras.layers.SpatialDropout2D(0.08, name=f"{name}_spatial_dropout")(x)


def build_attention_crnn_model(config: AudioConfig, class_count: int) -> tf.keras.Model:
    """Build a from-scratch residual CRNN with bidirectional sequence modeling and attention."""
    inputs = tf.keras.Input(
        (config.n_mels, config.frames_per_clip, 1), name="normalized_log_mel"
    )
    x = tf.keras.layers.Conv2D(
        32, 5, strides=(2, 1), padding="same", use_bias=False,
        kernel_initializer="he_normal", name="stem_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("swish", name="stem_swish")(x)

    # Downsample frequency and time gradually; keep 100+ temporal steps for the GRU.
    for name, filters, strides in (
        ("encoder_stage_1", 48, (1, 1)),
        ("encoder_stage_2", 80, (2, 2)),
        ("encoder_stage_3", 128, (2, 2)),
        ("encoder_stage_4", 160, (1, 1)),
    ):
        x = _crnn_residual_block(x, filters, strides=strides, name=name)

    # Pool only over frequency so each time frame remains available to the RNN.
    frequency_bins = (config.n_mels + 7) // 8
    average = tf.keras.layers.AveragePooling2D(
        pool_size=(frequency_bins, 1), name="frequency_average_pool"
    )(x)
    maximum = tf.keras.layers.MaxPooling2D(
        pool_size=(frequency_bins, 1), name="frequency_max_pool"
    )(x)
    sequence = tf.keras.layers.Concatenate(axis=-1, name="frequency_pool_fusion")(
        [average, maximum]
    )
    sequence = tf.keras.layers.Reshape((-1, 320), name="time_sequence")(sequence)
    sequence = tf.keras.layers.Dense(160, use_bias=False, name="frame_projection")(sequence)
    sequence = tf.keras.layers.LayerNormalization(name="frame_normalization")(sequence)
    sequence = tf.keras.layers.Activation("swish", name="frame_swish")(sequence)
    sequence = tf.keras.layers.Dropout(0.15, name="frame_dropout")(sequence)
    sequence = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(96, return_sequences=True, dropout=0.15),
        name="bidirectional_gru",
    )(sequence)

    attention = tf.keras.layers.Dense(96, activation="tanh", name="attention_hidden")(sequence)
    attention = tf.keras.layers.Dense(1, name="attention_score")(attention)
    attention = tf.keras.layers.Softmax(axis=1, name="temporal_attention_weights")(attention)
    attended = tf.keras.layers.Dot(axes=1, name="temporal_attention_pool")(
        [attention, sequence]
    )
    attended = tf.keras.layers.Flatten(name="attention_context")(attended)
    mean = tf.keras.layers.GlobalAveragePooling1D(name="temporal_mean_pool")(sequence)
    maximum = tf.keras.layers.GlobalMaxPooling1D(name="temporal_max_pool")(sequence)
    x = tf.keras.layers.Concatenate(name="temporal_statistics_fusion")(
        [attended, mean, maximum]
    )
    x = tf.keras.layers.Dense(
        192,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="genre_projection",
    )(x)
    x = tf.keras.layers.Dropout(0.4, name="genre_dropout")(x)
    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", name="genre_probabilities"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="musify_attention_crnn")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=2.5e-4, weight_decay=1e-4, clipnorm=1.0
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_model(
    config: AudioConfig, class_count: int, architecture: str = "multiscale"
) -> tf.keras.Model:
    """Select one of Musify's independently implemented, from-scratch models."""
    if architecture == "multiscale":
        return build_multiscale_model(config, class_count)
    if architecture == "crnn":
        return build_crnn_model(config, class_count)
    if architecture == "attention_crnn":
        return build_attention_crnn_model(config, class_count)
    raise ValueError(f"Unsupported training architecture: {architecture}")
