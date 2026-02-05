import os 
import json
import torch
import numpy as np
# 做出新的vocabulary


PW_WORD = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ "

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def build_voc(model="GPT2-Hacker-password-generator",WORD=PW_WORD) -> list:

    model_voc=load_json(f"../model/{model}/vocab.json")
    idx2char={k:v for k,v in model_voc.items() if k in WORD}
    keys=[] # vocabulary
    values=[] # vocabulary index
    for key,value in idx2char.items():
        keys.append(key)
        values.append(value)
    return [keys,values]
    


