from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv1D, Dense, Layer, LayerNormalization

class AminoAcidClassPredictor(Model):

    def __init__(self,
                 n_classes: int,
                 use_conv: bool = True) -> None:
                 
        super().__init__()

        if use_conv:
            self.predict_class = Sequential()
            self.predict_class.add(LayerNormalization())
            self.predict_class.add(Conv1D(128, 5, activation='relu', padding='same', use_bias=True))
            self.predict_class.add(Conv1D(n_classes, 3, activation=None, padding='same', use_bias=True))
        else:
            self.predict_class = Sequential()
            self.predict_class.add(LayerNormalization())
            self.predict_class.add(Dense(512,activation='relu'))
            self.predict_class.add(Dense(n_classes,activation=None))

    def call(self, inputs):
        logits = self.predict_class(inputs['encoder_output'])
        inputs['sequence_logits'] = logits
        return inputs