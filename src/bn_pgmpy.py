import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.inference import VariableElimination
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('./data/processed/transitions.csv')

category_cols = []
for col in df.columns:
    if col.endswith('_category'):
        category_cols.append(col)
#print(category_cols)

df_model = df[category_cols + ['machine_state', 'next_machine_state']].copy()
df_model[category_cols] = df_model[category_cols].fillna('None')

alarm_ids = set()
for col in df_model.columns:
    if col.endswith('_count_category'):
        alarm_id = col.replace('_count_category', '')
        alarm_ids.add(alarm_id)

#print(alarm_ids)

#edges
edges = []
edges.append((('machine_state'), ('next_machine_state')))
for alarm_id in alarm_ids:
    edges.append((f'{alarm_id}_count_category', 'next_machine_state'))
    edges.append((f'{alarm_id}_duration_category', 'next_machine_state'))
    
bn = DiscreteBayesianNetwork(edges)

#train-test split
train_size = int(len(df_model) * 0.8)
df_train = df_model.iloc[:train_size]
df_test = df_model.iloc[train_size:]

bn.fit(df_train)

for cpd in bn.get_cpds():
    print(cpd)
    
inference = VariableElimination(bn)

actual_states, predicted_states = [], []

#predicts on test set
for index in range(len(df_test)):
    row = df_test.iloc[index]
    
    #evidence dict
    evidence = {}
    evidence['machine_state'] = row['machine_state']
    for col in category_cols:
        evidence[col] = row[col]
        
    #posterior prob for next state
    result = inference.query(variables=['next_machine_state'], 
                             evidence=evidence, show_progress=False)
    
    states = result.state_names['next_machine_state']
    predicted_state = states[result.values.argmax()]
    
    actual_states.append(row['next_machine_state'])
    predicted_states.append(predicted_state)

#eval metrics
print(classification_report(actual_states, predicted_states))
print(confusion_matrix(actual_states, predicted_states))