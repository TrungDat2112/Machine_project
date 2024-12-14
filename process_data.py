import pandas as pd
data = pd.read_csv('ev_data.csv')
processed_data = data.copy()

q1 = processed_data['Total Energy Delivered'].quantile(0.25)
q3 = processed_data['Total Energy Delivered'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
processed_data = processed_data[
    (processed_data['Total Energy Delivered'] >= lower_bound) &
    (processed_data['Total Energy Delivered'] <= upper_bound)
]

processed_data.to_csv('processed_ev_data.csv', index=False)

print("Sample Processed Data:")
print(processed_data[['Charging Time', 'Average Power', 'SOC Charged', 'Total Energy Delivered']].head())
processed_data.info()