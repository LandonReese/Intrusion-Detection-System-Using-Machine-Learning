# Modified Graphical User Interface to show what kind of attack and 
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

# Create a dictionary to store the reports and confusion matrices for all three algorithms
report_dict = {}

def run_boost_algorithm():
    # Load the data set
    df = pd.read_csv("data/CICIDS2017_sample_km.csv")

    # Define the feature and target variables
    X = df.drop(["Label"], axis=1)
    y = df["Label"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the selected boost algorithm from the dropdown menu
    selected_boost = boost_menu.get()

    # Make predictions on the test data using the trained model
    # y_pred = model.predict(X_test)
    y_pred_LGBM = lgb.LGBMClassifier()
    y_pred_LGBM.fit(X_train, y_train)
    y_pred_LGBM = y_pred_LGBM.predict(X_test)

    y_pred_XGB = xgb.XGBClassifier()
    y_pred_XGB.fit(X_train, y_train)
    y_pred_XGB = y_pred_XGB.predict(X_test)

    y_pred_Cat = cbt.CatBoostClassifier(verbose=0)
    y_pred_Cat.fit(X_train, y_train)
    y_pred_Cat = y_pred_Cat.predict(X_test)
    
    # Modify string to match the algorithm
    typeOfAlgo = ""

    # Train the selected boost algorithm on the training data
    if selected_boost == "LightGBM":
        model = y_pred_LGBM
        typeOfAlgo = "LightGBM"
        report_dict[selected_boost] = (classification_report(y_test, model.predict(X_test)),
                                       confusion_matrix(y_test, model.predict(X_test)))

    elif selected_boost == "XGBoost":
        model = y_pred_XGB
        model.fit(X_train, y_train)
        typeOfAlgo = "XGBoost"
        report_dict[selected_boost] = (classification_report(y_test, model.predict(X_test)),
                                       confusion_matrix(y_test, model.predict(X_test)))
    elif selected_boost == "CatBoost":
        model = y_pred_Cat
        model.fit(X_train, y_train)
        typeOfAlgo = "CatBoost"
        report_dict[selected_boost] = (classification_report(y_test, model.predict(X_test)),
                                       confusion_matrix(y_test, model.predict(X_test)))


    # Generate a classification report and confusion matrix
    report_LGBM = classification_report(y_test, y_pred_LGBM)
    report_XGB = classification_report(y_test, y_pred_XGB)
    report_Cat = classification_report(y_test, y_pred_Cat)

    matrix_LGBM = confusion_matrix(y_test, y_pred_LGBM)
    matrix_XGB = confusion_matrix(y_test, y_pred_XGB)
    matrix_Cat = confusion_matrix(y_test, y_pred_Cat)


    # Create a new window to display the classification report and confusion matrix
    report_window = tk.Toplevel()
    report_window.title("Report for " + typeOfAlgo + " algorithm")
    
    # Add a dropdown menu for selecting the algorithm
    algorithm_menu = ttk.Combobox(report_window, values=["LightGBM", "XGBoost", "CatBoost"])
    algorithm_menu.set("Select Algorithm")
    algorithm_menu.pack(pady=10)

    # Add a Text widget to the new window to display the classification report
    report_text = tk.Text(report_window, height=20, width=60)
    
    report_text.pack(pady=10)

    def show_report():
        # Get the selected algorithm from the dropdown menu
        selected_algorithm = algorithm_menu.get()

        # Update the report text widget with the classification report and confusion matrix for the selected algorithm
        report_text.delete(1.0, tk.END)
        report_text.insert(tk.END, report_LGBM);
        report_text.insert(tk.END, report_XGB);
        report_text.insert(tk.END, report_Cat);

        if selected_algorithm == "LightGBM":
            report_text.insert(tk.END, "Confusion Matrix:\n" + str(report_dict[selected_algorithm][1]))
        elif selected_algorithm == "XGBoost":
            report_text.insert(tk.END, "Confusion Matrix:\n" + str(report_dict[selected_algorithm][1]))
        elif selected_algorithm == "CatBoost":
            report_text.insert(tk.END, "Confusion Matrix:\n" + str(report_dict[selected_algorithm][1]))
        # report_text.insert(tk.END, "Classification Report:\n" + report_dict[selected_algorithm][0] + "\n\n")
        # report_text.insert(tk.END, "Confusion Matrix:\n" + str(report_dict[selected_algorithm][1]))

    # Add a button to display the report and confusion matrix for the selected algorithm
    view_button = tk.Button(report_window, text="View Report", command=show_report)
    view_button.pack(pady=10)

    # Display the report and matrix in a pop-up window
    if selected_boost == "LightGBM":
        messagebox.showinfo("Classification Report and Confusion Matrix", matrix_LGBM)
        plt.figure(figsize=(6, 6))
        sns.heatmap(matrix_LGBM, annot=True, cmap="Blues")
    elif selected_boost == "XGBoost":
        messagebox.showinfo("Classification Report and Confusion Matrix", matrix_XGB)
        plt.figure(figsize=(6, 6))
        sns.heatmap(matrix_XGB, annot=True, cmap="Blues")
    elif selected_boost == "CatBoost":
        messagebox.showinfo("Classification Report and Confusion Matrix", matrix_Cat)
        plt.figure(figsize=(6, 6))
        sns.heatmap(matrix_Cat, annot=True, cmap="Blues")

        
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()



# Create the GUI window using Tkinter
root = tk.Tk()
root.geometry("400x200")
root.title("LCCDE Security System")

# Add a dropdown menu for selecting the boost algorithm
boost_menu = ttk.Combobox(root, values=["LightGBM", "XGBoost", "CatBoost"])
boost_menu.set("Select Boost Algorithm")
boost_menu.pack(pady=10)

# Add a run button for running the selected boost algorithm on the data set
run_button = tk.Button(root, text="Run Boost Algorithm", command=run_boost_algorithm)
run_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()