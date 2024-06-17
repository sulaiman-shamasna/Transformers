# import numpy as np
# import tensorflow_text as tf_text
import tensorflow as tf
# import pathlib
# from typing import Tuple
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# from check_shape import ShapeChecker
from preprocessing import *
from models import *


def masked_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the masked loss.
    
    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
    
    Returns:
        tf.Tensor: Computed masked loss.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_acc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the masked accuracy.
    
    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
    
    Returns:
        tf.Tensor: Computed masked accuracy.
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

# Assuming 'model' is already defined and created
model.compile(optimizer='adam',
              loss=masked_loss, 
              metrics=[masked_acc, masked_loss])

vocab_size = target_text_processor.vocabulary_size()

# Print expected loss and accuracy
print({
    "expected_loss": tf.math.log(tf.cast(vocab_size, tf.float32)).numpy(),
    "expected_acc": 1 / vocab_size
})

# Evaluate the model before training
model.evaluate(val_ds, steps=20, return_dict=True)



# Training the model with the EarlyStopping and ModelCheckpoint callbacks
history = model.fit(
    train_ds.repeat(), 
    epochs=50,
    steps_per_epoch=50,
    validation_data=val_ds,
    validation_steps=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
    ]
)

model.save('trained_models/big_model8')

# model = tf.keras.models.load_model('big_model')
