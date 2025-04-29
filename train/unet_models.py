import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate

def build_Unet(input_shape, depth, dropout_rate, weights=None, batch_norm=False, add_output_conv=False, l2_lambda=0):
    inputs = Input(shape=input_shape)
    x = inputs
    # Encoder
    skips = []
    for i in range(depth):
        x = Conv2D(2 ** (5 + i), 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = Conv2D(2 ** (5 + i), 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
        if batch_norm == 2:
            x = tf.keras.layers.BatchNormalization()(x)
        skips.append(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate * (1 + 0.25 * i))(x)
        x = MaxPooling2D(2, strides=2)(x)
    # Decoder
    for i in range(depth):
        x = Conv2D(2 ** (5 + depth - i), 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = Conv2D(2 ** (5 + depth - i), 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
        if batch_norm == 2:
            x = tf.keras.layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate * (1 + 0.25 * (depth - i)))(x)
        x = Conv2DTranspose(2 ** (5 + depth - i - 1), (2, 2), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
        x = concatenate([x, skips[depth - 1 - i]])
    x = Conv2D(2 ** 5, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = Conv2D(2 ** 5, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
    if batch_norm == 2:
        x = tf.keras.layers.BatchNormalization()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    # Add output convolutional layer
    if add_output_conv == 1 or add_output_conv == 3:
        x = Conv2D(2 ** 4, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
    if add_output_conv == 2 or add_output_conv == 3:
        x = Conv2D(2 ** 5, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
    # Create output layer
    x = Conv2D(1, 1, activation='sigmoid', name="output1")(x)

    model = Model(inputs=inputs, outputs=[x])
    if weights:
        model.load_weights(weights)
    return model


def attention_gate(x, g, inter_channel):
    """
    Implementación de Attention Gate para U-Net
    x: feature map de skip connection
    g: feature map del decoder
    """
    theta_x = tf.keras.layers.Conv2D(inter_channel, 1, strides=1, padding='same')(x)
    phi_g = tf.keras.layers.Conv2D(inter_channel, 1, strides=1, padding='same')(g)
    
    concat = tf.keras.layers.add([theta_x, phi_g])
    concat = tf.keras.layers.Activation('relu')(concat)
    
    psi = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same')(concat)
    psi = tf.keras.layers.Activation('sigmoid')(psi)
    
    return tf.keras.layers.multiply([x, psi])

def build_attention_unet(input_shape, output_channels=1):
    # Entrada
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Encoder
    # Bloque 1
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Bloque 2
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bloque 3
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bloque 4
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder con Attention Gates
    # Bloque 6
    up6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    att6 = attention_gate(conv4, up6, 512)
    concat6 = tf.keras.layers.concatenate([up6, att6])
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(concat6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    # Bloque 7
    up7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    att7 = attention_gate(conv3, up7, 256)
    concat7 = tf.keras.layers.concatenate([up7, att7])
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(concat7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    # Bloque 8
    up8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    att8 = attention_gate(conv2, up8, 128)
    concat8 = tf.keras.layers.concatenate([up8, att8])
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(concat8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    # Bloque 9
    up9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    att9 = attention_gate(conv1, up9, 64)
    concat9 = tf.keras.layers.concatenate([up9, att9])
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Capa de salida
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='sigmoid')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model