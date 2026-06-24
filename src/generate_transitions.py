import pandas as pd

df = pd.read_csv('./data/processed/alarm_features.csv')

#markov property
df['next_machine_state'] = df['machine_state'].shift(-1)

#as last row has no next state its dropped
df = df.dropna(subset=['next_machine_state'])

df.to_csv('./data/processed/transitions.csv', index=False)