import pandas as pd

from sklearn.preprocessing import LabelEncoder

def encode_label(feature):
    label_encoder = LabelEncoder()
    label_encoder.fit(feature)
    print(feature.name,label_encoder.classes_)
    return label_encoder.transform(feature)
    

dataset = pd.read_csv("dataset/creditors.csv")
df = pd.DataFrame(dataset)

for col in df.select_dtypes(include=['object']).columns.tolist():
    df[str(col)] = encode_label(df[str(col)])

df.to_csv("dataset/creditors_ready.csv", index=False)