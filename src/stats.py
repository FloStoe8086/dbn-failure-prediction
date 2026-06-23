import pandas as pd

df = pd.read_csv('./data/dataset_exercise.csv')
alarm_freq = df.groupby('alarm_id')['time_window'].nunique().sort_values(ascending=False)

#calculates cumulative sum
cumsum = alarm_freq.cumsum() / alarm_freq.sum() * 100

#shows how many alarms are responsible for how many percent of the total windows
for n in [5, 10, 15, 20, 30, 50]:
    print(f"Top {n}: {cumsum.iloc[n-1]:.1f}%")