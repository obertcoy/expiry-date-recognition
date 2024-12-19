import numpy as np
import tensorflow as tf
from constant import CHAR_LIST, CRNN_WEIGHTS, INDEX_TO_CHAR_DICT
from tensorflow.keras.backend import ctc_decode, get_value

class MapToSequenceLayer(tf.keras.layers.Layer):

    def __call__(self, inputs):

        # Input = (batch_size, height, width, channels) => Feature Map

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]

        # Output = (batch_size, width, height * channels) => sequence per columns

        outputs = tf.reshape(inputs, (batch_size, width, height * channels))

        return outputs
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[1] * input_shape[3])
    
    def get_config(self):
        config = super(MapToSequenceLayer, self).get_config()
        return config
    
class CTCLoss(tf.keras.losses.Loss):

    def __init__(self, name: str = 'CTCLoss') -> None:

        super(CTCLoss, self).__init__()
        self.name = name
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight= None) -> tf.Tensor:

        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss
    
    def get_config(self):
        config = super(CTCLoss, self).get_config()
        return config

def create_model( LR = 0.001):
    
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        tf.keras.layers.Conv2D(512, kernel_size=(2, 2), padding='valid', activation='relu'),
        MapToSequenceLayer(),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Dense(len(CHAR_LIST) + 1, activation='softmax')
    ])

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= LR), loss= CTCLoss())

    return model

def load_saved_model(input_shape=(64, 200, 1), LR= 0.001):
    
    model = create_model(input_shape=input_shape, LR=LR)
        
    model.load_weights(CRNN_WEIGHTS)
    
    return model

def decode_label(encoded) -> list:
        
    decoded = []

    for char_idx in encoded:
        
        if int(char_idx) != -1 and int(char_idx) != len(CHAR_LIST): # -1 -> _ in CTC and len(char_list) -> pad
            
            decoded.append(INDEX_TO_CHAR_DICT[char_idx])
                
    return decoded

def get_model_output(Y_pred):
    
    ctc_input_length = np.ones(Y_pred.shape[0]) * Y_pred.shape[1]
    output = get_value(ctc_decode(Y_pred, input_length= ctc_input_length, greedy= True)[0][0])
    pred_text = ''.join([char for char in decode_label(output)])
    
    return pred_text