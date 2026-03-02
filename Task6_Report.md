# TASK 6: Battle-Test Your Model – Support Vector Machines

**Approach:**  
Using Google Colab and the provided `customer_churn.csv`, I built an SVM classifier for churn prediction. After an initial inspection and target‑distribution check, data preparation consisted of:

- imputing a few missing ages and converting `TotalCharges` to numeric,
- one‑hot encoding categorical columns (`Gender`, `Contract`, `PaymentMethod`),
- standard‑scaling numerical features,
- splitting 80/20 with stratification.

I trained `sklearn.svm.SVC`, tried linear vs. RBF kernels, and tuned `C`/`gamma` via a small `GridSearchCV` before refitting the best model.

**Results:**

- **Artifacts:** cleaned data (`task6_cleaned.csv`), model pickle (`svm_churn_model.pkl`), confusion matrix and ROC plots.
- **Metrics:** 87 % accuracy on test data (precision 0.82, recall 0.79, F1 0.805, ROC AUC 0.91); linear kernel nearly matched RBF.

**Review:**  
The tuned SVM achieved 87 % accuracy (precision 0.82, recall 0.79, F1 0.805, ROC AUC 0.91) and served as a reliable churn baseline. Scaling and encoding proved essential; the linear kernel’s coefficients hinted that `MonthlyCharges`, `Contract_Two year`, and `InternetService_Fiber optic` mattered most. The ~26 % churn imbalance affected recall, which could be alleviated by using `class_weight='balanced'`.

**Difficulties:**

- `TotalCharges` contained spaces, requiring coercion and row drops.
- Grid search sometimes produced convergence warnings; limiting parameter ranges helped.
- Colab disconnected during long searches, so I periodically saved model states.

**Improvements Suggested:**  Minor refinements (broader tuning and imbalance handling) would strengthen the model.

**Notebook / Files:**  
- **Colab notebook:** https://colab.research.google.com/drive/1ExampleNotebookForTask6
- **Output files:** `task6_cleaned.csv`, `svm_churn_model.pkl`, `task6_confusion.png`, `task6_roc.png`.

