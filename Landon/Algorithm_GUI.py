# Modified Graphical User Interface to show what kind of attack and 
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
import catboost as cbt
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# Create a dictionary to store the reports and confusion matrices for all three algorithms
report_dict = {}

def run_boost_algorithm():
    # Load the data set
    df = pd.read_csv("Landon/CICIDS2017_sample_km.csv")

    # Define the feature and target variables
    X = df.drop(["Label"], axis=1)
    y = df["Label"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions on the test data using the trained model
    y_pred_LGBM = lgb.LGBMClassifier()
    y_pred_XGB = xgb.XGBClassifier()
    y_pred_Cat = cbt.CatBoostClassifier(verbose=0)

    # Train the selected boost algorithm on the training data
    y_pred_LGBM.fit(X_train, y_train)
    report_dict[0] = (classification_report(y_test, y_pred_LGBM.predict(X_test)),
                            confusion_matrix(y_test, y_pred_LGBM.predict(X_test)))

    y_pred_XGB.fit(X_train, y_train)
    report_dict[1] = (classification_report(y_test, y_pred_XGB.predict(X_test)),
                            confusion_matrix(y_test, y_pred_XGB.predict(X_test)))
   
    y_pred_Cat.fit(X_train, y_train)
    report_dict[2] = (classification_report(y_test, y_pred_Cat.predict(X_test)),
                            confusion_matrix(y_test, y_pred_Cat.predict(X_test)))

    # Show report function
    def show_report(report, matrix):
        messagebox.showinfo("Classification Report", report)
        plt.figure(figsize=(6, 6))
        sns.heatmap(matrix, annot=True, cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
        
    # Display the reports
    for i in range(3):
        # report_dict[i][0] is the classification report
        # report_dict[i][1] is the confusion matrix
        show_report(report_dict[i][0], report_dict[i][1])
    
    # Create a new window to display the classification report and confusion matrix
    report_window = tk.Toplevel()
    report_window.title("Report for all algorithms")

    # Add a Text widget to the new window to display the classification report
    report_text = tk.Text(report_window, height=20, width=60)
    report_text.pack(pady=10)

    for i in range(3):
        if i == 0:
            report_text.insert(tk.END, "LightGBM\n")
        elif i == 1:
            report_text.insert(tk.END, "XGBoost\n")
        elif i == 2:
            report_text.insert(tk.END, "CatBoost\n")
        report_text.insert(tk.END, "Classification Report:\n" + report_dict[i][0] + "\n\n")
        report_text.insert(tk.END, "Confusion Matrix:\n" + str(report_dict[i][1]) + "\n\n")

    # Add a button to close the window
    close_button = tk.Button(report_window, text="Close", command=report_window.destroy)
    close_button.pack(pady=10)

# Create the GUI window using Tkinter
root = tk.Tk()
root.geometry("400x200")
root.title("LCCDE Security System")

# Add a run button for running the selected boost algorithm on the data set
run_button = tk.Button(root, text="Run Boost Algorithm", command=run_boost_algorithm)
run_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()