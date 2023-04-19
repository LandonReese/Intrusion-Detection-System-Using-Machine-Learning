import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import catboost as cbt
import xgboost as xgb
import time
from river import stream
from statistics import mode
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random

df = pd.read_csv(r"C:\Users\evanm\Desktop\CICIDS2017_sample_km.csv")



df.Label.value_counts()

X = df.drop(['Label'],axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0) #shuffle=False

pd.Series(y_train).value_counts()

from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,sampling_strategy={2:1000,4:1000})

X_train, y_train = smote.fit_resample(X_train, y_train)

pd.Series(y_train).value_counts()


# Train the LightGBM algorithm
import lightgbm as lgb
lg = lgb.LGBMClassifier()

# Train the LightGBM algorithm
import lightgbm as lgb
lg = lgb.LGBMClassifier()

# Start the timer
start_time = time.time()

lg.fit(X_train, y_train)

# Stop the timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for LightGBM: {elapsed_time} seconds")

lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
print(classification_report(y_test,y_pred))
print("Accuracy of LightGBM: "+ str(accuracy_score(y_test, y_pred)))
print("Precision of LightGBM: "+ str(precision_score(y_test, y_pred, average='weighted')))
print("Recall of LightGBM: "+ str(recall_score(y_test, y_pred, average='weighted')))
print("Average F1 of LightGBM: "+ str(f1_score(y_test, y_pred, average='weighted')))
print("F1 of LightGBM for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))
lg_f1=f1_score(y_test, y_pred, average=None)

# Plot the confusion matrix
cm=confusion_matrix(y_test,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# Train the XGBoost algorithm
import xgboost as xgb
xg = xgb.XGBClassifier()

X_train_x = X_train.values
X_test_x = X_test.values

# Start the timer
start_time = time.time()

xg.fit(X_train_x, y_train)

# Stop the timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for XGBoost: {elapsed_time} seconds")

xg.fit(X_train_x, y_train)

y_pred = xg.predict(X_test_x)
print(classification_report(y_test,y_pred))
print("Accuracy of XGBoost: "+ str(accuracy_score(y_test, y_pred)))
print("Precision of XGBoost: "+ str(precision_score(y_test, y_pred, average='weighted')))
print("Recall of XGBoost: "+ str(recall_score(y_test, y_pred, average='weighted')))
print("Average F1 of XGBoost: "+ str(f1_score(y_test, y_pred, average='weighted')))
print("F1 of XGBoost for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))
xg_f1=f1_score(y_test, y_pred, average=None)

# Plot the confusion matrix
cm=confusion_matrix(y_test,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# Train the CatBoost algorithm
import catboost as cbt
cb = cbt.CatBoostClassifier(verbose=0,boosting_type='Plain')
#cb = cbt.CatBoostClassifier()
start_time = time.time()
cb.fit(X_train, y_train)
# Stop the timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for CatBoost: {elapsed_time} seconds")
y_pred = cb.predict(X_test)
print(classification_report(y_test,y_pred))
print("Accuracy of CatBoost: "+ str(accuracy_score(y_test, y_pred)))
print("Precision of CatBoost: "+ str(precision_score(y_test, y_pred, average='weighted')))
print("Recall of CatBoost: "+ str(recall_score(y_test, y_pred, average='weighted')))
print("Average F1 of CatBoost: "+ str(f1_score(y_test, y_pred, average='weighted')))
print("F1 of CatBoost for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))
cb_f1=f1_score(y_test, y_pred, average=None)

# Plot the confusion matrix
cm=confusion_matrix(y_test,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

model=[]
for i in range(len(lg_f1)):
    if max(lg_f1[i],xg_f1[i],cb_f1[i]) == lg_f1[i]:
        model.append(lg)
    elif max(lg_f1[i],xg_f1[i],cb_f1[i]) == xg_f1[i]:
        model.append(xg)
    else:
        model.append(cb)
        
model



def intrusion_description(label):
    descriptions = {
        0: "Normal",
        1: "DoS slowloris",
        2: "DoS Slowhttptest",
        3: "DoS Hulk",
        4: "DoS GoldenEye",
        # Add more descriptions based on the dataset labels
    }
    return descriptions.get(label, "Unknown")

def train_boosting_algo(algo):
    df = pd.read_csv(r"C:\Users\evanm\Desktop\CICIDS2017_sample_km.csv")

    X = df.drop(['Label'],axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0)

    # Train the selected algorithm
    if algo == "LightGBM":
        # Train LightGBM
        lg = lgb.LGBMClassifier()
        lg.fit(X_train, y_train)
        y_pred = lg.predict(X_test)
    elif algo == "XGBoost":
        # Train XGBoost
        xg = xgb.XGBClassifier()
        X_train_x = X_train.values
        X_test_x = X_test.values
        xg.fit(X_train_x, y_train)
        y_pred = xg.predict(X_test_x)
    elif algo == "CatBoost":
        # Train CatBoost
        cb = cbt.CatBoostClassifier(verbose=0,boosting_type='Plain')
        cb.fit(X_train, y_train)
        y_pred = cb.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
    print("Precision: " + str(precision_score(y_test, y_pred, average='weighted')))
    print("Recall: " + str(recall_score(y_test, y_pred, average='weighted')))
    print("F1: " + str(f1_score(y_test, y_pred, average='weighted')))

def intrusion_alert():
    # Select a random test sample
    index = random.randint(0, len(X_test) - 1)
    sample = X_test.iloc[index].values.reshape(1, -1)
    true_label = y_test.iloc[index]

    # Make predictions using the trained models
    lg_pred = int(lg.predict(sample)[0])
    xg_pred = int(xg.predict(sample)[0])
    cb_pred = int(cb.predict(sample)[0])

    # Get the majority vote
    preds = [lg_pred, xg_pred, cb_pred]
    predicted_label = max(set(preds), key=preds.count)

    # Get the intrusion description
    description = intrusion_description(predicted_label)

    # If it's an intrusion, block it and display a notification
    if predicted_label != 0:
        messagebox.showinfo(
            "Intrusion Blocked",
            f"Intrusion type: {description}\n\nDetailed information:\nPredicted label: {predicted_label}\n\nThis intrusion has been automatically blocked."
        )

# Schedule the intrusion_alert function to run periodically
def schedule_intrusion_alert():
    intrusion_alert()
    root.after(5000, schedule_intrusion_alert)  # Adjust the time (in milliseconds) between alerts as desired
root = tk.Tk()
root.title("Intrusion Detection System")
frame = ttk.Frame(root, padding="10 10 10 10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
# Replace the command for the alert_button in the GUI code
alert_button = ttk.Button(frame, text="Simulate Intrusion Alert", command=schedule_intrusion_alert)


# Function to periodically trigger the intrusion alert
def simulate_drive():
    while True:
        time.sleep(random.uniform(5, 15))  # Wait for a random interval between 5 and 15 seconds
        approved = intrusion_alert()
        if approved:
            print("Intrusion approved.")
        else:
            print("Intrusion denied.")

# Run the simulation
simulate_drive()


root = tk.Tk()
root.title("Intrusion Detection System")

# Use ttk to style the widgets
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)

frame = ttk.Frame(root, padding="10 10 10 10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

alert_button = ttk.Button(frame, text="Simulate Intrusion Alert", command=intrusion_alert)
alert_button.grid(row=0, column=0, pady=10)

root.mainloop()


import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
import time

# Function to train the selected model
def train_model():
    model = model_var.get()
    if model == "LightGBM":
        # Train LightGBM
        lg.fit(X_train, y_train)
        messagebox.showinfo("Training", "LightGBM training completed.")
    elif model == "XGBoost":
        # Train XGBoost
        xg.fit(X_train_x, y_train)
        messagebox.showinfo("Training", "XGBoost training completed.")
    elif model == "CatBoost":
        # Train CatBoost
        cb.fit(X_train, y_train)
        messagebox.showinfo("Training", "CatBoost training completed.")
    else:
        messagebox.showerror("Error", "Please select a valid boosting algorithm.")
        
def simulate_intrusion_alert():
    # Code to simulate intrusion alert
    pass

# Function to periodically trigger the intrusion alert
def simulate_drive():
    while True:
        time.sleep(random.uniform(5, 15))  # Wait for a random interval between 5 and 15 seconds
        approved = intrusion_alert()
        if approved:
            print("Intrusion approved.")
        else:
            print("Intrusion denied.")


root = tk.Tk()
root.title("Intrusion Detection System")

frame = ttk.Frame(root, padding="10 10 10 10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create a dropdown menu to select the boosting algorithm
model_var = tk.StringVar()
model_var.set("Select Algorithm")
model_dropdown = ttk.OptionMenu(frame, model_var, "Select Algorithm", "LightGBM", "XGBoost", "CatBoost")
model_dropdown.grid(row=0, column=0, pady=10)

# Button to train the selected model
train_button = ttk.Button(frame, text="Train Model", command=train_model)
train_button.grid(row=1, column=0, pady=10)

# Button to simulate driving
simulate_drive_button = ttk.Button(frame, text="Simulate Drive", command=lambda: root.after(5000, schedule_intrusion_alert))
simulate_drive_button.grid(row=2, column=0, pady=10)

root.mainloop()
