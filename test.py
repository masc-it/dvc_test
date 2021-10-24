import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers, models,  callbacks
import tensorflow as tf
import json
import os
from ruamel.yaml import YAML
np.random.seed(1337)

def load_params():
    "Updates FULL_PARAMS with the values in params.yaml and returns all as a dictionary"
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)
    return params

params = load_params()


dataset = pd.read_csv("dataset/creditors_test.csv")
df = pd.DataFrame(dataset)

labels = df["class"]
data_ = df.drop("class", axis=1)

test_set = np.asarray(data_)

test_labels = np.asarray(labels)

scaler = StandardScaler()

test_set = scaler.fit_transform(test_set)

metrics =[]
def build_model():
  model = models.Sequential()
  model.add(layers.Dense(20, input_dim=20, activation='relu'))
  #model.add(layers.Dropout(0.1))
  model.add(layers.Dense(params["model"]["units1"], activation='relu'))

  model.add(layers.Dense(params["model"]["units1"], activation='relu'))
  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(30, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer=params["model"]["optimizer"], metrics=['accuracy'])
  return model
	
model = build_model()

model.load_weights("weights/best_val_loss.h5")
acc = model.evaluate(test_set, test_labels, batch_size=params["train"]["batch_size"])

obj = {"accuracy": acc[1]}

with open('metrics/results_test.json', 'w') as outfile:
  json.dump(obj, outfile)