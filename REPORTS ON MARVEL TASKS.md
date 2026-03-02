# Report on Tasks Completed

### TASK 1: MATLAB Machine Learning Onramp Course

Completed the free online MATLAB Machine Learning Onramp (approx. 3–4 hours) to learn the basics of data import, visualization, and model training within MATLAB’s environment. Worked through exercises on linear regression, classification, and neural network apps using synthetic datasets.

Key takeaways:
- Got hands‑on with MATLAB syntax (arrays, tables, `fitcsvm`, etc.).
- Learned to use the built‑in data preprocessing and performance reporting tools.
- Faced occasional Wi‑Fi drops and struggled initially with MATLAB’s command‑history interface.

Overall, the course provided a gentle introduction to model workflows and reinforced concepts I later applied in Python.
![MATLAB Onramp Certificate](Screenshot%202025-10-18%20005011.png)
---

### TASK 2: Kaggle Crafter – Build & Publish Your Own Dataset

Built a synthetic dataset of 100 student records in Google Colab using the Faker library (fields: name, age, grade, major, GPA, email) and exported to CSV. Uploaded it to Kaggle with a descriptive README and appropriate tags, learning about dataset licensing and metadata requirements.

Challenges included defining realistic field distributions and ensuring no duplicates; selecting the right topic area helped with discoverability.

**Dataset:** [Fake Student Information Dataset](https://www.kaggle.com/datasets/jayaramlnaik/fake-student-information-dataset)

---

### TASK 3: Data Detox - Data Cleaning using Pandas

Loaded `customer_data.csv` into a Pandas DataFrame in Colab and performed extensive cleaning: `.drop_duplicates()`, regex‑based typo correction, `pd.to_datetime` for date fields, outlier clipping on ages, and `.fillna()`/`.dropna()` decisions. Created functions to validate ranges and used `info()`/`describe()` to audit progress.

Saved the cleaned result as `cleaned_customer_data.csv` and documented steps in the notebook.

**Notebook:** https://colab.research.google.com/drive/1cEB_NKEtyk0m-GRk_2V94zLoVVCkAA-D

---

### TASK 4: Anomaly Detection

Processed G‑Flix log data (user ID, session length, pages viewed, errors) with NumPy/Pandas, normalized features, and computed Z‑scores to flag values beyond 3σ. Also trained an `IsolationForest` from Scikit‑Learn (contamination=0.01) and compared its outlier labels to the Z‑score list. Produced side‑by‑side tables and plotted anomalies with Matplotlib.

Both methods identified a similar set of suspicious accounts; I reported the top five users for each and discussed differences.

**Notebook:** https://colab.research.google.com/drive/1tv9gm37Wof47HKB6srdrA5T44OgW0Wdz

---

### TASK 5: Logistic Regression from Scratch

Used the Framingham heart disease dataset (`framingham.csv`) to implement logistic regression manually in Python. Steps included feature scaling with mean/variance, adding a bias term, defining the sigmoid function, and iterating gradient descent updates to minimize binary cross‑entropy loss. After training, compared the hand‑rolled classifier’s coefficients and predictions to `sklearn.linear_model.LogisticRegression`.

Accuracy, precision, recall and F1 matched within a few percentage points of Scikit‑Learn’s version, validating the implementation and deepening my understanding of logistic curves and decision boundaries.

**Notebook:** [Logistic_Regression.ipynb](https://github.com/JayaramNaik/marvel-level2/blob/main/Logistic_Regression.ipynb)

---

### TASK 6: Battle-Test Your Model – Support Vector Machines

Built an SVM in Colab on `customer_churn.csv` after preprocessing:
- imputed missing `Age` with median,
- encoded categorical columns (`Gender`, `Contract`, `PaymentMethod`) via one‑hot or label encoding,
- scaled numeric features with `StandardScaler`,
- performed an 80/20 stratified train/test split.

Ran a simple grid search over kernels (`linear`, `rbf`, `poly`) and C values; best model (rbf, C=1) yielded 87 % accuracy (precision 0.82, recall 0.79, F1 0.805, AUC 0.91). Feature importance gleaned from an `SVC` with `coef_` suggested monthly charges and contract type were most predictive. Class imbalance (~74 % non‑churn) harmed recall; trying `class_weight='balanced'` improved it to 0.83.

**Artifacts:** cleaned data CSV, serialized model pickle, confusion matrix and ROC plots in notebook.  
**Notebook:** https://colab.research.google.com/drive/1BJnOEuvBM9mUtWj37j72V7wpPTGYGadO  
**Notes:** dealt with messy `TotalCharges` field (some blanks), convergence warnings, and occasional Colab disconnects; further tuning and SMOTE remain options.

---

### TASK 7: Fairness Evaluation – ID3 Decision Tree on Utrecht Recruitment Data

Implemented the ID3 algorithm from scratch on the Utrecht hiring dataset, handling missing gender entries and bucketing age into decades. Capped the tree depth at 5 to prevent overfitting and trained on 80 % of the data.

Evaluated fairness by computing hire rates, true positive rates, and demographic parity differences across gender and age groups. The model achieved 0.82 accuracy (F1 0.79) but showed bias: hired males 52 % vs females 47 %, and applicants 35+ were hired 58 % compared to 39 % for those under 25. These gaps violated demographic parity and equal opportunity; removing age reduced skew but lowered overall accuracy.

**Artifacts:** `id3_uftree.py` implementation and analysis notebook; fairness metrics were computed manually.  
**Notebook:** https://colab.research.google.com/drive/1-0o8bqsiXV0fDQ_d1OKy-sDrUT3E6ZNM  
**Note:** future work should exclude protected attributes and integrate an automated audit routine.

---

### TASK 8: KNN Classifier with Feature Ablation Study

Using the Breast Cancer Wisconsin dataset, converted data to a DataFrame, dropped any `id` column, encoded target (`M`→1, `B`→0), and normalized all 30 features with `StandardScaler` because KNN is scale‑sensitive. Performed an 80/20 stratified split.

Trained `KNeighborsClassifier(n_neighbors=5)` and logged baseline metrics (accuracy 0.95, precision 0.94, recall 0.96, F1 0.95). For ablation, iterated over features: removed one column, refitted scaler and model, and evaluated on the original test set. Recorded results in `ablation_results.csv`; losing `worst radius` dropped accuracy to 0.90 (F1 0.91), while eliminating `mean symmetry` changed metrics by <0.5 %.

**Artifacts:** ablation script, CSV, and Matplotlib plot of metric change; scaling refits slowed the loop.  
**Notebook:** https://colab.research.google.com/drive/1n76_0fAetIJTETb6PboyhKl11cfMITgq  
**Note:** trying other k values or recursive feature elimination could yield deeper insights.

---

### TASK 9: Model Evaluation – Comparing Pretrained Pickle Files

Loaded five `.pkl` models (three classifiers, two regressors) and applied them to a common test dataset after replicating preprocessing (scaling, one‑hot encoding) used during training. Used `joblib.load` and inspected each model’s class to choose metrics.

For classifiers computed accuracy, precision, recall, F1 via `classification_report`; for regressors computed RMSE and R². Results were written to `model_comparison.csv` and visualized with side‑by‑side bar charts. Handled mismatches by reordering test DataFrame columns and imputing NaNs; one model needed an older Scikit‑Learn version to load without warnings.

**Highlights:**
- Best classifier: random forest (F1 ≈ 0.86); SVM and logistic trailed.
- Best regressor: gradient‑boosted (RMSE ≈ 2.15) vs linear (RMSE ≈ 3.68).
- Preprocessing consistency was the main challenge; automated pipelines would prevent errors.

**Artifacts:** evaluation script, comparison CSV, classification/regression plots.  
**Notebook:** https://colab.research.google.com/drive/1fha_v2rJ0X1JweTzK22iHjRfNUD51nwI  
**Note:** wrapping the flow in reusable functions made reruns easy when new models arrived.

---
