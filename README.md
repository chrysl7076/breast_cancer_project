CODTECH IT SOLUTIONS INTERNSHIP 5TH JUNE 2025 → 5TH JULY 2025

TASK 4 – Create a predictive model using scikit learn to classify or predict outcomes from a dataset(eg. Spam mail detection).

DELIVERABLE: a Juypter notebook showcasing the model’s implementation and evaluation

NAME - CHRYSL SHECKINA THOMAS
INTERN ID - CT04DG328
DOMAIN - PYTHON PROGRAMMING
DURATION - 4 WEEKS 
MENTOR - MS. NEELA SANTHOSH KUMAR

DESCRIPTION BELOW SCREENSHOTS

OUTPUT SCREENSHOTS

![alt text](images/Screenshot%202025-06-27%20230054.png)
![alt text](images/Screenshot%202025-06-27%20230116-1.png)
![alt text](images/Screenshot%202025-06-27%20230132.png)
![alt text](images/Screenshot%202025-06-27%20230146.png)
![alt text](images/Screenshot%202025-06-27%20230159.png)
![alt text](images/Screenshot%202025-06-27%20230213-1.png)
![alt text](images/Screenshot%202025-06-27%20230227.png)
![alt text](images/Screenshot%202025-06-27%20230239-1.png)
![alt text](images/Screenshot%202025-06-27%20230255.png)
![alt text](images/Screenshot%202025-06-27%20230308.png)
![alt text](images/Screenshot%202025-06-27%20230319.png)
![alt text](images/Screenshot%202025-06-27%20230330.png)
![alt text](images/Screenshot%202025-06-27%20230339.png)

DESCRIPTION
This project focuses on building a predictive machine learning model to detect breast cancer based on real diagnostic data. Using the Breast Cancer Wisconsin(Diagnostic) Data Set from Kaggle, the model aims to classify tumors as either benign(non-cancerous) or malignant(cancerous). The task falls under binary classification, which is a fundamental use in healthcare- focused machine learning. 

This project was developed using Python, Jupyter Notebook, and scikit-learn library. It is a part of CodTech Internship Task that requires creating a complete predictive model pipeline – from data loading and cleaning, to training, evaluating, and comparing different models. 

The Dataset used was obtained from Kaggle. It contains 569 samples of Breast Cancer diagnostic measurements with 30 numericasl features computed from digitized images of fine needle aspirate(FNA) of breast masses. 

Each sample is labelled as either:
 M(Malignant) – Cancerous
B(Benign) – Non Cancerous

Key Columns: radius mean, texture mean, area mean, etc. 

Diagnosis: The notebook is structured step by step to reflect a real world ML pipeline. 

Data Loading:
The Dataset(data.csv) is read into a pandas DataFrame and the first few rows are inspected. 

Data Cleaning and Preprocessing 
Unnecessary columns such as ‘id’ and empty columns were dropped.

The target column ‘diagnosis’ was converted from categorical (“M”,”B”) to binary values(1,0). Featured labels were split. The dataset was divided into training (80%) and testing(20%) sets. 

StandardScaler was applied to normalize feature values. 

Model Building: A Random Forest Classifier was initially selected for it s robustness and high performance on classification problems. The model was trained using the training dataset. 

Model Evaluation Predictions were made on the test set, and the evaluation metrics like accuracy, confusion matrix, and the classification report were used. The model achieved ovr 97% accuracy, indicating excellent predictive performance. 

Model Comaparison:
Two more models – Logistic Regression and Support Vector Machine(SVM) were added and trained on the same data. Their performances were compared using accuracy scores and visualized in a bar chart. 

ROC Curve & AUCScore to further evaluate the models, ROC curves were plotted and the AUC scores were calculated. All models achieved AUC>0.98, confirming that they effectively distinguish between malignant and benign cases. 

Technologies used:
 
Python 3.13.2
Jupyter Notebook 
Pandas
NumPy
Sci-kit Learn
Matplotlib
Seaborn

This project demonstrated the full lifecycle of building a predictive model using real world healthcare data. Among the models tested, Random Forest Classifier provide the best overall performance in terms of accuracy and AUC. The visualizations like ROC curves and confusion matrices helped in clearly interpreting model performance. 

With proper tuning and deployment, such a model could serve as a valuable tool in assisting medical professionals with early diagnosis of breast cancer. However it is important to note that machine learning models are meant to support, not replace clinical decision making. 

Project Files
- breast_cancer_model.ipynb -- Full notebook with code and outputs
- data.csv -- Dataset used from Kaggle
- README.md -- Project description and instructions

