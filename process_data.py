import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# Normalize skewed distributions using log transformation
processed_data['Charging Time'] = np.log1p(processed_data['Charging Time'])
processed_data['Total Energy Delivered'] = np.log1p(processed_data['Total Energy Delivered'])

# Scale features to standardize them
scaler = StandardScaler()
processed_data[['Charging Time', 'Average Power', 'SOC Charged']] = scaler.fit_transform(
    processed_data[['Charging Time', 'Average Power', 'SOC Charged']]
)

# Save processed data to a new CSV file
processed_data.to_csv('processed_ev_data.csv', index=False)

# Display sample processed data
print("Sample Processed Data:")
print(processed_data[['Charging Time', 'Average Power', 'SOC Charged', 'Total Energy Delivered']].head())
processed_data.info()