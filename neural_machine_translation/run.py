# from preprocessing import *
from models import *
from train import *


import textwrap
print('Expected output:\n', '\n'.join(textwrap.wrap(target_raw[-1])))

inputs = [
    'Hace mucho frio aqui.', # "It's really cold here."
    'Esta es mi vida.', # "This is my life."
    'Su cuarto es un desastre.' # "His room is a mess"
]

for t in inputs:
    print(model.translate([t])[0].numpy().decode())

    print()
    result = model.translate(inputs)

    print(result[0].numpy().decode())
    print(result[1].numpy().decode())
    print(result[2].numpy().decode())
    print()