# Report on Tasks Completed

### TASK 1: MATLAB Machine Learning Onramp Course

**Approach:**  
I enrolled in the MATLAB Machine Learning Onramp course and completed the interactive lessons to learn the basics of machine learning using MATLAB. I focused on understanding how to handle data, train models, and evaluate their performance within the MATLAB environment.

**Review:**  
The course was well-organized and helped me get hands-on experience with machine learning concepts. The exercises were clear and helped me learn step-by-step.

**Difficulties:**

- Sometimes the page would reset due to internet issues, causing loss of progress.
- Getting used to MATLAB’s syntax took some extra time since I’m not familiar with that language.

**Improvements Suggested:**

- It would be helpful to have downloadable materials for offline study.
 
 ### Screenshot — Course Completion

![image](<Screenshot 2025-10-18 005011.png>)
---

### TASK 2: Kaggle Crafter – Build & Publish Your Own Dataset

**Approach:**  
I created a synthetic dataset of 100 student records using Python’s Faker library in Google Colab. The dataset includes details like student ID, name, age, gender, marks, and grades. I uploaded the dataset to Kaggle, added all necessary metadata like description, tags, license, and file details.

**Review:**  
This task helped me understand how to prepare a dataset that is clean, well-documented, and ready to be shared publicly. I learned the importance of metadata and clear documentation to make datasets usable and trustworthy.

**Difficulties:**

- I faced a logic problem in the dataset: the marks and grades do not match correctly. For example, one student scored 100 marks but got a grade ‘C’, while another scored 78 marks but got an ‘A’. This inconsistency reduces the credibility of the dataset.
- Choosing the right tags on Kaggle was confusing because some of the tags I wanted were not available.

**Dataset (Kaggle):** [Fake Student Information Dataset](https://www.kaggle.com/datasets/jayaramlnaik/fake-student-information-dataset)

---

### TASK 3: Data Detox - Data Cleaning using Pandas

**Approach:**  
I worked on cleaning the customer_data.csv dataset in the Copy_of_Datadetox.ipynb notebook. I started by loading the data and removing exact duplicate rows. Then, I fixed typos in categorical columns like Country, Gender, and PreferredDevice using replace() with dictionaries. I handled impossible values in Age and TotalPurchase by setting them to NaN if outside logical ranges (e.g., Age <0 or >100). I converted SignupDate and LastLogin to datetime, fixed temporal inconsistencies where LastLogin was before SignupDate, and imputed missing values: numerical columns with median, categorical with mode, and placeholders for unique fields like Email. Finally, I dropped rows with missing CustomerID or SignupDate, filtered out fake names (numeric entries), and saved the cleaned dataset as cleaned_customer_data.csv.

**Review:**  
This task was an excellent hands-on experience with real-world data cleaning challenges. I learned the importance of inspecting data thoroughly, using appropriate imputation strategies, and preserving data integrity. The step-by-step approach helped me understand Pandas methods for handling duplicates, inconsistencies, formatting, and missing values, preparing the data effectively for analysis or modeling.

**Difficulties:**  
Understanding the logic for temporal checks (e.g., LastLogin before SignupDate) took some thought, as did choosing between dropping vs. imputing missing values. Handling dates with errors='coerce' was new, and ensuring no nulls remained required careful verification.

**Improvements Suggested:**  
The dataset could include more diverse data types for broader practice. Adding automated validation checks or using libraries like Great Expectations for data quality could enhance the process. The tutorial format was helpful, but more emphasis on why certain strategies (e.g., median vs. mean) are chosen would be beneficial.

---

### TASK 4: Anomaly Detection

**Approach:**  
I worked on detecting anomalies in the G-Flix user activity logs using the anomaly_detection.csv dataset in the Anomaly_detection.ipynb notebook. I started by loading the data and performing basic inspection with info() and describe() to identify red flags like extreme max values. I created visualizations including box plots and histograms with KDE for login_duration_min, data_accessed_MB, and files_downloaded to understand normal behavior trends. For statistical anomaly detection, I applied Z-score on the numerical features, flagging users with Z-score > 3 as suspects. For unsupervised ML, I used Isolation Forest on scaled data to detect multivariate anomalies. I compared the results to find high-confidence suspects flagged by both methods, visualized them on a scatter plot, and prepared a final report with the top 5 suspects based on data accessed and evidence.

**Review:**  
This task provided practical experience in anomaly detection for security forensics. I learned to combine statistical methods (Z-score) with ML (Isolation Forest) for robust detection, and the importance of scaling data and visualizing results. The step-by-step approach helped differentiate outliers from genuine anomalies, and building the investigative report improved storytelling skills.

**Difficulties:**  
Choosing the right contamination parameter for Isolation Forest and interpreting multivariate anomalies was tricky. Ensuring the Z-score threshold (3) was appropriate for the dataset required understanding statistical significance.

**Improvements Suggested:**  
The dataset could include more features for deeper analysis. Adding evaluation metrics like precision or using other algorithms (e.g., DBSCAN) could enhance comparison. The tutorial style was effective, but more guidance on tuning hyperparameters would be helpful.

---

### TASK 5: Logistic Regression from Scratch

**Approach:**  
I implemented logistic regression from scratch and compared it with Scikit-Learn using the framingham.csv dataset in the Logistic_Regression.ipynb notebook. I started by loading the data, exploring it with info(), checking missing values, target distribution, and a correlation heatmap. For preprocessing, I dropped rows with missing values, manually scaled features (standardization), added a bias column, and split into train/test sets. The scratch implementation included sigmoid, log loss, gradient descent for fitting, and prediction functions. I trained the model, visualized the loss curve, calculated metrics (accuracy, precision, recall, F1) from scratch, and created a confusion matrix heatmap. Finally, I implemented using Scikit-Learn, compared metrics, and visualized the differences.

**Review:**  
This task deepened my understanding of logistic regression mechanics, matrix operations, and gradient descent. Building from scratch showed the inner workings, while comparing with Scikit-Learn highlighted abstraction benefits. I learned to handle class imbalance, evaluate models properly, and the importance of scaling and preprocessing.

**Difficulties:**  
Implementing gradient descent and ensuring convergence without vanishing gradients was challenging. Handling the bias term in matrix form and debugging the loss function took time. Class imbalance affected recall, requiring careful metric interpretation.

**Improvements Suggested:**  
The dataset could include more balanced classes or techniques like SMOTE. Adding regularization to the scratch model would make it more robust. The step-by-step breakdown was excellent, but more on hyperparameter tuning (e.g., learning rate) would help.

