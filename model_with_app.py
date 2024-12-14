import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class EnergyDNN(nn.Module):
    def __init__(self, input_size):
        super(EnergyDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

class EnergyPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Energy Consumed Predict")
        self.root.geometry("600x400")
        
        self.model_linear = joblib.load('linear_model.pkl')
        self.rf_model = joblib.load('rf_model.pkl')
        self.gb_model = joblib.load('gb_model.pkl')
        self.ridge_model = joblib.load('ridge_model.pkl')
        self.lasso_model = joblib.load('lasso_model.pkl')
        self.svr_model = joblib.load('svr_model.pkl')

        self.model_nn = EnergyDNN(3)  
        self.model_nn.load_state_dict(torch.load('energy_dnn_model.pth', weights_only=True))
        self.model_nn.eval()
        
        self.current_model = None
        
        self.model_label = tk.Label(self.root, text="Choose Model:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10)
        
        self.model_select = ttk.Combobox(self.root, values=["LinearRegression", "RandomForestRegressor", "GradientBoosting", "Ridge", "Lasso","SVR", "DNN"])
        self.model_select.grid(row=0, column=1, padx=10, pady=10)
        self.model_select.set("LinearRegression") 
        
        self.input_fields = {}
        input_labels = ['Charging Time', 'Average Power', 'SOC Charged']
        
        for i, label in enumerate(input_labels):
            lbl = tk.Label(self.root, text=label)
            lbl.grid(row=i + 1, column=0, padx=10, pady=5, sticky="e")
            entry = tk.Entry(self.root)
            entry.grid(row=i + 1, column=1, padx=10, pady=5)
            self.input_fields[label] = entry
        
        self.generate_button = tk.Button(self.root, text="Generate", command=self.generate_prediction)
        self.generate_button.grid(row=len(input_labels) + 1, column=0, columnspan=2, pady=20)
        
        self.result_label = tk.Label(self.root, text="Prediction will be shown here", font=("Arial", 14))
        self.result_label.grid(row=len(input_labels) + 2, column=0, columnspan=2)
    
    def generate_prediction(self):
        inputs = [float(self.input_fields[label].get()) for label in self.input_fields]
        
        model_choice = self.model_select.get()
        if model_choice == "LinearRegression":
            model = self.model_linear
        elif model_choice == "RandomForestRegressor":
            model = self.rf_model
        elif model_choice == "GradientBoosting":
            model = self.gb_model
        elif model_choice == "Ridge":
            model = self.ridge_model
        elif model_choice == "Lasso":
            model = self.lasso_model
        elif model_choice == "SVR":
            model = self.svr_model
        elif model_choice == "DNN":
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            output = self.model_nn(inputs_tensor).item()
            self.result_label.config(text=f"Predicted Total Energy Delivered: {output:.2f}")
            return
        
        inputs_scaled = np.array(inputs).reshape(1, -1)
        
        if model_choice != "DNN":
            scaler = StandardScaler()
            inputs_scaled = scaler.fit_transform(inputs_scaled)  
            
        prediction = model.predict(inputs_scaled)[0]
        self.result_label.config(text=f"Predicted Total Energy Delivered: {prediction:.2f}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyPredictionApp(root)
    root.mainloop()
