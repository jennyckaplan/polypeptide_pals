import AminoAcidClassPredictor
import tensorflow as tf

class SequenceToSequenceClassificationTask(Task):

    def __init__(self,
                 key_metric: str,
                 deserialization_func: Callable[[bytes], Dict[str, tf.Tensor]],
                 n_classes: int,
                 label_name: str,
                 input_name: str = 'encoder_output',
                 output_name: str = 'sequence_logits',
                 mask_name: str = 'sequence_mask'):
        super().__init__(key_metric, deserialization_func)
        self._n_classes = n_classes
        self._label_name = label_name
        self._input_name = input_name
        self._output_name = output_name
        self._mask_name = mask_name

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        labels = inputs[self._label_name]
        logits = outputs[self._output_name]
        if self._mask_name != 'sequence_mask':
            mask = outputs[self._mask_name]
        else:
            #mask = rk.utils.convert_sequence_length_to_sequence_mask(labels, inputs['protein_length'])
            mask = tf.sequence_mask(labels, maxlen=inputs['protein_length'])
        loss, accuracy = classification_loss_and_accuracy(
            labels, logits, mask)
        metrics = {self.key_metric: accuracy}
        return loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(AminoAcidClassPredictor(self._n_classes, self._input_name, self._output_name))
        return layers
