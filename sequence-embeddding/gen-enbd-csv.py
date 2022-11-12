import pandas as pd
import json

READ_LIMIT = 5000
EMBEDDINGS = []

for i in range(READ_LIMIT):
    with open(f'./embeddings/{i}.txt', 'r') as f:
        EMBEDDINGS.append(json.loads(f.read()))


data = pd.read_csv(
    './prepd-dataset.csv', dtype={'opcode': object})

data = data.iloc[:5000]

labels = data['swc_label'].tolist()

data = pd.DataFrame(zip(EMBEDDINGS, labels),
                    columns=['embeddings', 'swc_label'])

print(data)

# .merge(data, right_index=True, left_index=True)
features = data.embeddings.apply(pd.Series)
data = features.merge(data, right_index=True, left_index=True)
data = data.drop(['embeddings'], axis=1)

print(data)

data.to_csv('./embeddings.csv', index=False)
