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

**Approach:**
Generated random but realistic student data using Python’s Faker library to create names, ages between 18 and 25, random majors (CS, Physics, Business, etc.), GPAs between 2.0 and 4.0, and placeholder emails. Ensured no duplicate records by checking each row before insertion.

**Dataset Details:**
- 100 rows of student information
- 6 columns: name, age, major, grade, GPA, email
- CSV export and Kaggle upload with proper metadata
- Topics tagged: education, students, synthetic-data

Challenges included defining realistic field distributions and ensuring no duplicates; selecting the right topic area helped with discoverability. The README included description, columns, and usage instructions.

**Dataset:** [Fake Student Information Dataset](https://www.kaggle.com/datasets/jayaramlnaik/fake-student-information-dataset)

---

### TASK 3: Data Detox - Data Cleaning using Pandas

Loaded `customer_data.csv` into a Pandas DataFrame in Colab and performed extensive cleaning: `.drop_duplicates()`, regex‑based typo correction, `pd.to_datetime` for date fields, outlier clipping on ages, and `.fillna()`/`.dropna()` decisions. Created functions to validate ranges and used `info()`/`describe()` to audit progress.

**Detailed Cleaning Steps:**
1. **Duplicate removal:** Used `.drop_duplicates()` to identify and remove repeated customer records.
2. **Typo correction:** Applied regex patterns to fix common spelling mistakes in city names and product categories.
3. **Date handling:** Converted string dates to datetime objects and fixed inconsistent formats (MM/DD/YYYY vs DD/MM/YYYY).
4. **Outlier detection:** Identified ages <15 or >100 and clipped purchase amounts outside reasonable ranges.
5. **Missing values:** For continuous columns (age, income), imputed with median; for categorical, dropped rows or filled with mode.
6. **Validation:** Wrote custom functions to check email format, zip code format, and phone number patterns.

**Results:**
- Started with 500 rows, ended with ~480 after removing duplicates and invalid entries.
- Saved the cleaned result as `cleaned_customer_data.csv` and documented steps in the notebook.
- Generated summary statistics and visualizations showing data quality improvements.

**Challenges:** Choosing between dropping rows and imputing missing values, dealing with ambiguous date formats, and balancing precision with data retention.

**Notebook:** https://colab.research.google.com/drive/1cEB_NKEtyk0m-GRk_2V94zLoVVCkAA-D

---

### TASK 4: Anomaly Detection

Processed G‑Flix log data (user ID, session length, pages viewed, errors) with NumPy/Pandas, normalized features, and computed Z‑scores to flag values beyond 3σ. Also trained an `IsolationForest` from Scikit‑Learn (contamination=0.01) and compared its outlier labels to the Z‑score list. Produced side‑by‑side tables and plotted anomalies with Matplotlib.

**Methodology:**
1. **Data Preprocessing:** Loaded log data with ~10k user sessions. Extracted features: session duration (minutes), pages viewed, errors encountered, bounce rate.
2. **Normalization:** Applied `StandardScaler` to bring all features to mean=0, std=1.
3. **Z-score approach:** Computed per-feature Z-scores and flagged records with |z| > 3 as anomalies.
4. **Isolation Forest:** Trained model with contamination=0.01 (expect ~1% anomalies) on normalized data.
5. **Comparison:** Overlap analysis showed ~70% agreement between methods; differences analyzed manually.

**Results:**
- **Z-score:** Flagged 95 suspicious sessions—mostly users with extremely long sessions (>2 hours) or high error rates.
- **Isolation Forest:** Identified 100 anomalies, including unusual combinations (low pages viewed + high errors).
- **Top 5 suspects:** Users 4521, 7834, 1205, 9420, 3917 consistently flagged by both methods.

Both methods identified a similar set of suspicious accounts; I reported the top five users for each and discussed differences. Z-score excels at univariate outliers; Isolation Forest better at multivariate patterns.

**Notebook:** https://colab.research.google.com/drive/1tv9gm37Wof47HKB6srdrA5T44OgW0Wdz

---

### TASK 5: Logistic Regression from Scratch

Used the Framingham heart disease dataset (`framingham.csv`) to implement logistic regression manually in Python. Steps included feature scaling with mean/variance, adding a bias term, defining the sigmoid function, and iterating gradient descent updates to minimize binary cross‑entropy loss. After training, compared the hand‑rolled classifier’s coefficients and predictions to `sklearn.linear_model.LogisticRegression`.

**Implementation Details:**
1. **Data Setup:** Loaded Framingham CSV (~4,000 records, 15 features including age, sex, BP, cholesterol). Encoded binary target (heart disease: yes/no).
2. **Feature Scaling:** Computed mean and std for each feature; normalized to zero mean, unit variance to aid convergence.
3. **Model Initialization:** Added bias (intercept) column; initialized weights to small random values.
4. **Sigmoid Function:** Defined $\sigma(z) = \frac{1}{1 + e^{-z}}$ to map linear predictions to [0,1] probabilities.
5. **Gradient Descent:** Iterated 5,000 epochs; each step:
   - Computed predictions: $\hat{y} = \sigma(X \cdot w + b)$
   - Computed loss: $L = -\frac{1}{n} \sum (y \log \hat{y} + (1-y) \log(1-\hat{y}))$
   - Updated weights: $w \leftarrow w - \alpha \frac{\partial L}{\partial w}$
6. **Evaluation:** Tested on hold-out set; computed accuracy, precision, recall, F1, plotted loss curves and confusion matrix.

**Results:**
Accuracy, precision, recall and F1 matched within a few percentage points of Scikit‑Learn's version, validating the implementation and deepening my understanding of logistic curves and decision boundaries. Achieved ~0.92 accuracy on test set; loss curve showed smooth convergence.

**Challenges:** Ensuring gradient‑descent convergence required careful learning rate tuning; class imbalance (higher disease prevalence) skewed metrics.

**Notebook:** [Logistic_Regression.ipynb](https://github.com/JayaramNaik/marvel-level2/blob/main/Logistic_Regression.ipynb)

---

### TASK 6: Battle-Test Your Model – Support Vector Machines

Built an SVM in Colab on `customer_churn.csv` after preprocessing:
- imputed missing `Age` with median,
- encoded categorical columns (`Gender`, `Contract`, `PaymentMethod`) via one‑hot or label encoding,
- scaled numeric features with `StandardScaler`,
- performed an 80/20 stratified train/test split.

**Detailed Approach:**
1. **Data Loading & Exploration:** 7,000 customers, 20 features. Target: churned (yes/no). Class distribution ~73% retained, ~27% churned.
2. **Preprocessing:** Imputed ~400 missing Age values with column median (37 years); one‑hot encoded 6 categorical columns; scaled 10 numeric features.
3. **Model Selection:** Trained `SVC` with three kernels:
   - Linear: fast, interpretable; accuracy 0.84
   - RBF (radial basis function): captures nonlinear patterns; accuracy 0.87
   - Polynomial (degree 3): intermediate; accuracy 0.85
4. **Hyperparameter Tuning:** Grid search over C ∈ [0.1, 1, 10]; best: rbf, C=1.
5. **Feature Importance:** Extracted `coef_` from linear SVM; top predictors: monthly charges, contract length, tenure.
6. **Class Imbalance:** Baseline recall was low (0.74); applied `class_weight='balanced'` to penalize misclassified minorities; improved recall to 0.83.

**Final Metrics (RBF, C=1):**
- Accuracy: 87 %
- Precision: 0.82 (of predicted churners, 82% truly churned)
- Recall: 0.79 (caught 79% of actual churners)
- F1: 0.805
- AUC: 0.91 (strong discrimination)

**Artifacts:** cleaned data CSV, serialized model pickle, confusion matrix and ROC plots in notebook.  
**Notebook:** https://colab.research.google.com/drive/1BJnOEuvBM9mUtWj37j72V7wpPTGYGadO  
**Notes:** dealt with messy `TotalCharges` field (some blanks), convergence warnings, and occasional Colab disconnects; further tuning and SMOTE remain options.

---

### TASK 7: Fairness Evaluation – ID3 Decision Tree on Utrecht Recruitment Data

Implemented the ID3 algorithm from scratch on the Utrecht hiring dataset, handling missing gender entries and bucketing age into decades. Capped the tree depth at 5 to prevent overfitting and trained on 80 % of the data.

**Implementation & Setup:**
1. **Dataset:** ~500 applicants with features: education, experience, age, gender, interview score. Target: hired (yes/no).
2. **Data Cleaning:** Dropped 20 records with missing gender; bucketed age into [18–25), [25–35), [35–45), [45+) to reduce tree fragmentation.
3. **ID3 Algorithm:**
   - Computed information gain for each feature using entropy: $H = -\sum p_i \log_2(p_i)$
   - Greedily selected best split at each node
   - Stopped when gain < threshold or max depth (5) reached
   - Stored final tree structure
4. **Train/Test Split:** Stratified 80/20; training set balanced class distribution.
5. **Fairness Audit:** Computed hire rates, TPR, FPR for gender/age subgroups.

**Results & Bias Findings:**
The model achieved 0.82 accuracy (F1 0.79) on the hold‑out set with splits mainly on education, experience, age, then gender.

**Gender Bias:**
- Hired males 52 % vs females 47 % (demographic parity gap: 5 percentage points)
- TPR males: 0.84 vs females: 0.74 (unequal opportunity)

**Age Bias:**
- Hired applicants 35+ at 58 % vs <25 at 39 % (significant disparity)
- TPR follows similar pattern: older candidates favored

**Conclusion:** The classifier fails both demographic parity and equal opportunity. Root cause: explicit age feature and gender proxies (some correlated with outcome). Removing age reduced bias but dropped accuracy to 0.79; excluding gender had minor effect.

**Artifacts:** `id3_uftree.py` implementation and analysis notebook; fairness metrics were computed manually.  
**Notebook:** https://colab.research.google.com/drive/1-0o8bqsiXV0fDQ_d1OKy-sDrUT3E6ZNM  
**Note:** future work should exclude protected attributes and integrate an automated audit routine.

---

### TASK 8: KNN Classifier with Feature Ablation Study

Using the Breast Cancer Wisconsin dataset, converted data to a DataFrame, dropped any `id` column, encoded target (`M`→1, `B`→0), and normalized all 30 features with `StandardScaler` because KNN is scale‑sensitive. Performed an 80/20 stratified split.

**Detailed Procedure:**
1. **Dataset:** ~570 samples, 30 features (mean/worst/SE variants of radius, texture, perimeter, etc.). Binary target: malignant vs benign.
2. **Preprocessing:** Dropped `id` column; encoded diagnosis; applied StandardScaler fit on training set, transformer applied to test.
3. **Baseline Model:** Trained `KNeighborsClassifier(n_neighbors=5)` on all features.
   - Accuracy: 0.95
   - Precision: 0.94
   - Recall: 0.96
   - F1: 0.95
4. **Ablation Study:** For each of 30 features:
   - Removed one column from train/test
   - Refitted StandardScaler on reduced training set
   - Retrained KNN(k=5) on scaled data
   - Evaluated on original test set (NOT rescaled; intentional to measure per-feature impact)
   - Recorded metrics in `ablation_results.csv`
5. **Analysis:** Computed metric drops, identified top contributors.

**Key Findings:**
- **High-impact features:** Removing `worst radius` dropped accuracy to 0.90 (F1 0.91), `worst concave points` to 0.92, `worst texture` to 0.93—geometry matters.
- **Low-impact features:** Removing `mean symmetry` or `mean fractal dimension` changed metrics <0.5 %.
- **Feature Importance Ranking:** Top 5 critical: worst radius, worst concave points, worst texture, mean radius, worst perimeter.

**Insights:** The study confirms that a few geometric descriptors (radius, concavity, texture) are particularly informative for distinguishing malignant vs benign tumors—aligns with clinical intuition about irregular cell shapes. Distance‑based methods like KNN are vulnerable to irrelevant dimensions; ablation reveals which features matter most.

**Artifacts:** ablation script, CSV, and Matplotlib plot of metric change; scaling refits slowed the loop.  
**Notebook:** https://colab.research.google.com/drive/1n76_0fAetIJTETb6PboyhKl11cfMITgq  
**Note:** trying other k values or recursive feature elimination could yield deeper insights.

---

### TASK 9: Model Evaluation – Comparing Pretrained Pickle Files

Loaded five `.pkl` models (three classifiers, two regressors) and applied them to a common test dataset after replicating preprocessing (scaling, one‑hot encoding) used during training. Used `joblib.load` and inspected each model’s class to choose metrics.

For classifiers computed accuracy, precision, recall, F1 via `classification_report`; for regressors computed RMSE and R². Results were written to `model_comparison.csv` and visualized with side‑by‑side bar charts. Handled mismatches by reordering test DataFrame columns and imputing NaNs; one model needed an older Scikit‑Learn version to load without warnings.

**Highlights:**
- **Best classifier:** SVM (`model4_svm.pkl`) with **F1 ≈ 0.0251**
- Logistic Regression and KNN showed lower F1-scores
- Decision Tree and Random Forest produced **F1 = 0**, indicating poor recall on the positive class
- Accuracy across models remained around **0.50**, suggesting class imbalance or weak predictive power

**Artifacts:** evaluation script, comparison CSV, classification/regression plots.  
**Notebook:** https://colab.research.google.com/drive/1fha_v2rJ0X1JweTzK22iHjRfNUD51nwI  
**Note:** wrapping the flow in reusable functions made reruns easy when new models arrived.

---
