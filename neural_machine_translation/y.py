from train import model
import tensorflow as tf

inputs = [
    'Hace mucho frio aqui.', # "It's really cold here."
    'Esta es mi vida.', # "This is my life."
    'Su cuarto es un desastre.', # "His room is a mess"
    'Hola.'
]

class Export(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
  def translate(self, inputs):
    return self.model.translate(inputs)

export = Export(model)

_ = export.translate(tf.constant(inputs))

result = export.translate(tf.constant(inputs))

print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()

tf.saved_model.save(export, 'trained_models/translator',
                    signatures={'serving_default': export.translate})


reloaded = tf.saved_model.load('trained_models/translator')
_ = reloaded.translate(tf.constant(inputs)) #warmup

result = reloaded.translate(tf.constant(inputs))

print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()