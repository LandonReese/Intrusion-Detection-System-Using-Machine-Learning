import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
from tkinter import ttk


# Load the dataset
df = pd.read_csv(r"C:\Users\evanm\Desktop\CICIDS2017_sample_km.csv")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Label', axis=1), df['Label'], test_size=0.2, random_state=42)

# Function to plot feature importances
def plot_feature_importance(importance, model_name, features):
    plt.figure(figsize=(10, 5))
    plt.bar(features, importance)
    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=90)
    plt.show()
    
def plot_roc_curve(fpr, tpr, model_name):
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"{model_name} ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.show()

def plot_precision_recall_curve(precision, recall, model_name):
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, label=f"{model_name} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.legend()
    plt.show()

# Create the GUI window using Tkinter
root = tk.Tk()
root.geometry("400x200")
root.title("Boost Algorithm Selector")

progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=200, mode='determinate')
progress.pack(pady=10)

# Add a button for training XGBoost on the dataset
def train_xgb():
    
    
    progress.start()

    # Train the XGBoost algorithm
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # Make predictions on the test data and display a report and a confusion matrix
    y_pred = xgb_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    messagebox.showinfo("XGBoost Report", f"Report:\n{report}\n\nConfusion Matrix:\n{cm}")

    # Plot feature importance
    plot_feature_importance(xgb_model.feature_importances_, "XGBoost", X_train.columns)
    progress.stop()

xgb_button = tk.Button(root, text="Train XGBoost", command=train_xgb)
xgb_button.pack(pady=10)

# Add a button for training LightGBM on the dataset
def train_lgb():
    
    
    progress.start()
    # Train the LightGBM algorithm
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(X_train, y_train)

    # Make predictions on the test data and display a report and a confusion matrix
    y_pred = lgb_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    messagebox.showinfo("LightGBM Report", f"Report:\n{report}\n\nConfusion Matrix:\n{cm}")

    # Plot feature importance
    plot_feature_importance(lgb_model.feature_importances_, "LightGBM", X_train.columns)
    progress.stop()

lgb_button = tk.Button(root, text="Train LightGBM", command=train_lgb)
lgb_button.pack(pady=10)

# Add a button for training CatBoost on the dataset
def train_cb():
    
    
    progress.start()
    # Train the CatBoost algorithm
    cb_model = cbt.CatBoostClassifier(verbose=0)
    cb_model.fit(X_train, y_train)

    # Make predictions on the test data and display a report and a confusion matrix
    y_pred = cb_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    messagebox.showinfo("CatBoost Report", f"Report:\n{report}\n\nConfusion Matrix:\n{cm}")

    # Plot feature importance
    plot_feature_importance(cb_model.feature_importances_, "CatBoost", X_train.columns)
    progress.stop()

cb_button = tk.Button(root, text="Train CatBoost", command=train_cb)
cb_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()