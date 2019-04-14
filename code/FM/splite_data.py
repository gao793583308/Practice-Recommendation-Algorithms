import pandas as pd
data = pd.read_table("../../data/final_track2_train.txt", header=None)


train_data = data.sample(frac=0.99, axis=0)
test_data = data[~data.index.isin(train_data.index)]

train_data.to_csv("../../data/train_data.txt", sep="\t", header=None, index=None)
test_data.to_csv("../../data/test_data.txt", sep="\t", header=None, index=None)