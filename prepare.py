import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encode_label(feature):
    label_encoder = LabelEncoder()
    label_encoder.fit(feature)
    print(feature.name,label_encoder.classes_)
    return label_encoder.transform(feature)
    

dataset = pd.read_csv("dataset/creditors.csv")
df = pd.DataFrame(dataset)

for col in df.select_dtypes(include=['object']).columns.tolist():
    df[str(col)] = encode_label(df[str(col)])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train, test = train_test_split(df, test_size=0.2)

train.to_csv("dataset/creditors_train.csv", index=False)
test.to_csv("dataset/creditors_test.csv", index=False)