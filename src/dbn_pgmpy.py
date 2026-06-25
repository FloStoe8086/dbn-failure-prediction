import pandas as pd
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
from pgmpy.inference import VariableElimination

#https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/models/DynamicBayesianNetwork.py 
#https://pgmpy.org/started/quickstart.html

df = pd.read_csv('./data/processed/transitions.csv')

#keeps only category, machine state and next machine state columns
category_cols = []
for col in df.columns:
    if col.endswith('_category'):
        category_cols.append(col)
#print(category_cols)

df_model = df[category_cols + ['machine_state', 'next_machine_state']].copy()

#("Variable name", "timestep")
new_column_structure = []
for col in df_model.columns:
    if col == 'machine_state':
        new_column_structure.append(('machine_state', 0))
    elif col == 'next_machine_state':
        new_column_structure.append(('machine_state', 1))
    else:
        new_column_structure.append((col, 0))

df_model.columns = pd.MultiIndex.from_tuples(new_column_structure)
#print(df_model.columns)


alarm_ids = set()
for col in df_model.columns:
    #tuple requires []
    if col[0].endswith('_count_category'):
        alarm_id = col[0].replace('_count_category', '')
        alarm_ids.add(alarm_id)

#print(alarm_ids)

#edges
edges = []

edges.append((('machine_state', 0), ('machine_state', 1)))

for alarm_id in alarm_ids:
    #pgmpy requires to define (_category, 1) otherwise KeyError
    df_model[(f'{alarm_id}_count_category', 1)] = df_model[(f'{alarm_id}_count_category', 0)].values
    df_model[(f'{alarm_id}_duration_category', 1)] = df_model[(f'{alarm_id}_duration_category', 0)].values
    
    edges.append(((f'{alarm_id}_count_category', 0), ('machine_state', 1)))
    edges.append(((f'{alarm_id}_duration_category', 0), ('machine_state', 1)))

#print(edges)

dbn = DBN()

dbn.add_edges_from(edges)

# for edge in dbn.edges:
#     print(edge)

#print(df_model.columns)

#learn ctps
dbn.fit(df_model, estimator='MLE')

for cpd in dbn.cpds:
    print(cpd)

#db_inference = DBNInference(dbn)


    









