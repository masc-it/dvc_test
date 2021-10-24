from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer 
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers, models, optimizers, utils, callbacks, backend, regularizers
import tensorflow as tf
import json
import os
from ruamel.yaml import YAML
np.random.seed(1337)

metrics = []
val_metrics = []

def load_params():
    "Updates FULL_PARAMS with the values in params.yaml and returns all as a dictionary"
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)
    return params

params = load_params()

class CustomCallback(callbacks.Callback):
    ep_ = 1
    def on_epoch_begin(self, epoch, logs=None):
        print("epoch started")

    def on_epoch_end(self, epoch, logs=None):
        print("epoch ended")
        
        self.ep_ += 1
        with open('metrics/metrics.json', 'w') as outfile:
          json.dump(metrics, outfile)
  
    def on_train_batch_end(self, batch, logs=None):
        if batch % 2 == 0:
            metrics.append({"accuracy": logs["accuracy"]})
    
    def on_test_end(self, logs=None):
        
      val_metrics.append({"accuracy": logs["accuracy"]})


def scheduler(epoch, lr):
  lr_ = lr
  if epoch % 5 == 0:
    lr_ = lr_ * tf.math.exp(-0.10)

  return lr_

weights_path = "weights"
if not os.path.exists(weights_path):
  os.makedirs(weights_path)

cs = [
      callbacks.ModelCheckpoint(filepath=weights_path + "/best_val_loss.h5" ,
                                      monitor='val_loss',
                                      mode='min', save_weights_only=True, save_best_only=True),
      callbacks.ModelCheckpoint(filepath=weights_path + "/best_loss.h5" ,
                                      monitor='loss',
                                      mode='min', save_weights_only=True, save_best_only=True),
      callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
      callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_delta=1e-4, min_lr=0.0001),
      CustomCallback(),
      callbacks.LearningRateScheduler(scheduler, verbose=1)


  ]

dataset = pd.read_csv("dataset/creditors_ready.csv")
df = pd.DataFrame(dataset)

labels = df["class"]
data_ = df.drop("class", axis=1)

training_set, test_set, training_labels, test_labels = train_test_split(data_, labels, test_size=0.20)

training_set = np.asarray(training_set)
test_set = np.asarray(test_set)

training_labels = np.asarray(training_labels)
test_labels = np.asarray(test_labels)

print(training_labels.shape)
print(test_labels.shape)

print(np.count_nonzero(training_labels == 0))
print(np.count_nonzero(test_labels == 0))

scaler = StandardScaler()

training_set = scaler.fit_transform(training_set)

test_set = scaler.fit_transform(test_set)

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

model.fit(training_set, training_labels, validation_split=0.1, epochs=params["train"]["epochs"], batch_size=params["train"]["batch_size"], callbacks=[cs])

obj = metrics[-1]
obj["val_accuracy"] = val_metrics[-1]["accuracy"]

with open('metrics/results.json', 'w') as outfile:
  json.dump(obj, outfile)
