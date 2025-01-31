import torch

ckpt = torch.load('/Users/arasvalizadeh/Downloads/BArdiya-Barbod/bardia_data.pt' , map_location= 'cpu')

print(len(ckpt['input_data']))

