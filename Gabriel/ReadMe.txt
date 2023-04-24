Intrusion Detection System using LCCDE
This program demonstrates an intrusion detection system using the LightGBM, XGBoost, and CatBoost machine learning algorithms combined with the LCCDE (Leader-based Classifier Confidence-based Dynamic Ensemble) method.

Requirements
To run this program, you will need the following libraries:

pandas
numpy
matplotlib
seaborn
sklearn
lightgbm
catboost
xgboost
imbalanced-learn
river

You can install these libraries using pip but you may run into problems using pip to install river. We used Anaconda as our runtime enviorment and succesfully
installed all the libraries through that channel:


How to Run
Prepare your dataset: Make sure that your dataset is in CSV format and is titled "CICIDS2017_sample_km.csv" Replace the file path in the following code 
line with the path to the dataset:

Example: Line 15: df = pd.read_csv(r"C:\Users\gtroc\Jupyter Projects\CICIDS2017_sample_km.csv")

Execute the program: Run the entire script in your Python environment (e.g., Jupyter Notebook, Spyder, or any other IDE). The script will automatically load the dataset, preprocess it, train the models, and evaluate their performance using the LCCDE method.

Analyze the results: The program will print and display in a gui the performance metrics (accuracy, precision, recall, and F1-score) for each model and the LCCDE ensemble. It will also display confusion matrices for each model, which can help in understanding the classification performance.

Running the Poisoned Dataset: To run the poisoned dataset, simply change the name of the CSV file in Line 15 to "updated_CICIDS2017_sample_km_poisoned.csv"

Note: The program may take a considerable amount of time to run, depending on the size of your dataset and the processing power of your machine. It's recommended to run this program on a machine with sufficient memory and processing capabilities.