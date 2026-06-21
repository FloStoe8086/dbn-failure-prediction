import pandas as pd

df = pd.DataFrame(pd.read_excel('./data/dataset_exercise.xlsx'))
df.to_csv('./data/dataset_exercise.csv', index=False)