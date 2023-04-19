import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt

# Load the dataset
df = pd.read_csv(r"C:\Users\evanm\Desktop\CICIDS2017_sample_km.csv")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Label', axis=1), df['Label'], test_size=0.2, random_state=42)

# Create the GUI window using Tkinter
root = tk.Tk()
root.geometry("400x200")
root.title("Boost Algorithm Selector")

# Add a button for training XGBoost on the dataset
def train_xgb():
    # Train the XGBoost algorithm
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # Make predictions on the test data and display a report and a confusion matrix
    y_pred = xgb_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    messagebox.showinfo("XGBoost Report", f"Report:\n{report}\n\nConfusion Matrix:\n{cm}")

xgb_button = tk.Button(root, text="Train XGBoost", command=train_xgb)
xgb_button.pack(pady=10)

# Add a button for training LightGBM on the dataset
def train_lgb():
    # Train the LightGBM algorithm
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(X_train, y_train)

    # Make predictions on the test data and display a report and a confusion matrix
    y_pred = lgb_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    messagebox.showinfo("LightGBM Report", f"Report:\n{report}\n\nConfusion Matrix:\n{cm}")

lgb_button = tk.Button(root, text="Train LightGBM", command=train_lgb)
lgb_button.pack(pady=10)

# Add a button for training CatBoost on the dataset
def train_cb():
    # Train the CatBoost algorithm
    cb_model = cbt.CatBoostClassifier()
    cb_model.fit(X_train, y_train)

    # Make predictions on the test data and display a report and a confusion matrix
    y_pred = cb_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    messagebox.showinfo("CatBoost Report", f"Report:\n{report}\n\nConfusion Matrix:\n{cm}")

cb_button = tk.Button(root, text="Train CatBoost", command=train_cb)
cb_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()