import numpy as np
import tensorflow_text as tf_text
import tensorflow as tf
import pathlib
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from check_shape import ShapeChecker
from preprocessing import *
from models import *
from train import *


import textwrap

print('Expected output:\n', '\n'.join(textwrap.wrap(target_raw[-1])))

# # model.plot_attention(long_text)

inputs = [
    'Hace mucho frio aqui.', # "It's really cold here."
    'Esta es mi vida.', # "This is my life."
    'Su cuarto es un desastre.' # "His room is a mess"
]

# %%time
for t in inputs:
    print(model.translate([t])[0].numpy().decode())

    print()

    # %%time
    result = model.translate(inputs)

    print(result[0].numpy().decode())
    print(result[1].numpy().decode())
    print(result[2].numpy().decode())
    print()