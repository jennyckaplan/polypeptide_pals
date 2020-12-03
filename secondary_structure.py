from preprocess import deserialize_secondary_structure

from .Task import SequenceToSequenceClassificationTask


class SecondaryStructureTask(SequenceToSequenceClassificationTask):

    def __init__(self, num_classes: int = 8):
        assert num_classes >= 3
        assert num_classes <= 8
        super().__init__(
            key_metric='ACC', 
            deserialization_func=deserialize_secondary_structure, 
            n_classes=num_classes,
            label_name='secondary structure {num_classes}'.format(num_classes),
            input_name='encoder_output', output_name='sequence_logits')
