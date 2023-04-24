import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
import catboost as cbt
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def run_boost_algorithm():
    # Load the data set
    selected_dataset = dataset_menu.get()
    X, y = load_dataset(selected_dataset)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the selected boost algorithm from the dropdown menu
    selected_boost = boost_menu.get()

    # Train the selected boost algorithm on the training data
    if selected_boost == "LightGBM":
        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
    elif selected_boost == "XGBoost":
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
    elif selected_boost == "CatBoost":
        model = cbt.CatBoostClassifier(verbose=0)
        model.fit(X_train, y_train)
    elif selected_boost == "Hyperparam-tuned LightGBM":
        best_params = {'num_leaves': 50, 'n_estimators': 100, 'max_depth': 20, 'learning_rate': 0.1}
        model = lgb.LGBMClassifier(objective='multiclass', random_state=42, **best_params)
        model.fit(X_train, y_train)

    # Make predictions on the test data using the trained model
    y_pred = model.predict(X_test)

    # Generate a classification report and confusion matrix
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

        # Display the report and matrix in a pop-up window
    messagebox.showinfo("Classification Report", report)
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Create the GUI window using Tkinter
root = tk.Tk()
root.geometry("400x250")
root.title("LCCDE Security System")

# Add a dropdown menu for selecting the dataset
dataset_menu = ttk.Combobox(root, values=[r"C:\Users\evanm\Desktop\CICIDS2017_sample_km.csv", r"C:\Users\evanm\Desktop\updated_CICIDS2017_sample_km_poisoned.csv"])

dataset_menu.set("Select Dataset")
dataset_menu.pack(pady=10)

# Add a dropdown menu for selecting the boost algorithm
boost_menu = ttk.Combobox(root, values=["LightGBM", "XGBoost", "CatBoost", "Hyperparam-tuned LightGBM"])
boost_menu.set("Select Boost Algorithm")
boost_menu.pack(pady=10)

# Add a run button for running the selected boost algorithm on the data set
run_button = tk.Button(root, text="Run Boost Algorithm", command=run_boost_algorithm)
run_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()
