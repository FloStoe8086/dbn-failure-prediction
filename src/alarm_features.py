import pandas as pd

df = pd.read_csv('./data/dataset_exercise.csv')

#all_alarms = df.groupby('alarm_id')['time_window'].nunique().sort_values(ascending=False)
#counts unique time windows for each alarm_id
top20_alarms = df.groupby('alarm_id')['time_window'].nunique().sort_values(ascending=False).head(20)
print(top20_alarms)
#print(all_alarms)
#print(df[['start_alarm', 'end_alarm']].head(3).to_string())

result = pd.DataFrame()

#converts to list for easier iteration
top20_alarm_ids = top20_alarms.index.tolist()

#aggregates for each window 
for window in df['time_window'].unique():
    window_data = df[df['time_window'] == window]

    for alarm_id in top20_alarm_ids:
        alarm_data = window_data[window_data['alarm_id'] == alarm_id]

        #counts how often alarm_id appears in time window
        alarm_counter = len(alarm_data)
        result.loc[window, f'{alarm_id}_count'] = alarm_counter

        #sums duration of alarm_id in time window
        if len(alarm_data) > 0:
            start = pd.to_datetime(alarm_data['start_alarm'])
            end = pd.to_datetime(alarm_data['end_alarm'])
            duration_total = (end-start).dt.total_seconds().sum()
            result.loc[window, f'{alarm_id}_duration'] = duration_total 
        else:
            result.loc[window, f'{alarm_id}_duration'] = 0

    #determines machine state 
    result.loc[window, 'machine_state'] = 'Failure' if 'Failure' in window_data['machine_state'].values else 'Running'

#for NaN values fill with 0
result = result.fillna(0) 

def category_count(x):
    if x == 0:
        return 'None'
    elif x <= 2:
        return 'Low'
    elif x <= 5:
        return 'Medium'
    else:
        return 'High'

def category_duration(x):
    if x == 0:
        return 'None'
    elif x <= 30:
        return 'Short'
    elif x <= 300:
        return 'Medium'
    else:
        return 'Long'

#adds categories for count and duration for each alarm_id
for alarm_id in top20_alarm_ids:
    result[f'{alarm_id}_count_category'] = result[f'{alarm_id}_count'].apply(category_count) 
    result[f'{alarm_id}_duration_category'] = result[f'{alarm_id}_duration'].apply(category_duration)

#moves machine states to end of df
state_col = result.pop('machine_state')
result['machine_state'] = state_col

result.index.name = 'time_window'
result = result.reset_index()

result.to_csv('./data/processed/alarm_features.csv', index=False)