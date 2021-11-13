import os
import json


data = [1, 7, 10, 13, 16, 20, 23, 26, 29, 32, 36, 39, 42, 45, 48, 52, 55, 58, 61, 64, 67]
length = len(data)
for k in range(length):
  name = f'resnet21_layers_{k+1}.json'
  with open(name, 'w') as f:
    f.write(str(data))
  data = data[1:] + data[0]
