import pandas as pd

def merge_data(covid_data, stock_data, events):
    merged_data = pd.merge(covid_data, stock_data, left_on='date', right_index=True, how='inner')
    
    merged_data = add_event_dummy_variables(merged_data, events)
    
    return merged_data

def add_event_dummy_variables(data, events):
    data['Dummy_Variable'] = 0  
    
    for event_name, event_date in events.items():
        data.loc[data['date'] == event_date, 'Dummy_Variable'] = 1
    
    return data
