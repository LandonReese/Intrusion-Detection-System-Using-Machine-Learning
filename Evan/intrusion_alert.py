import tkinter as tk
from tkinter import ttk
from lccde import train_boosting_algo, simulate_intrusion_alert

# Define the GUI interface
root = tk.Tk()
root.title("Intrusion Detection System")

frame = ttk.Frame(root, padding="10 10 10 10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

model_var = tk.StringVar()
model_var.set("")

model_label = ttk.Label(frame, text="Select a boosting algorithm:")
model_label.grid(row=0, column=0, pady=10)

model_menu = ttk.OptionMenu(frame, model_var, "", "LightGBM", "XGBoost", "CatBoost")
model_menu.grid(row=0, column=1, pady=10)

train_button = ttk.Button(frame, text="Train Model", command=lambda: lccde.train_boosting_algo(model_var.get()))
train_button.grid(row=0, column=2, pady=10)

alert_button = ttk.Button(frame, text="Simulate Intrusion Alert", command=lambda: lccde.simulate_intrusion_alert())
alert_button.grid(row=1, column=0, pady=10, columnspan=3)

root.mainloop()