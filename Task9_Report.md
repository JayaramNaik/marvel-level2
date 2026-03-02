
# TASK 9: Model Evaluation – Comparing Pretrained Pickle Files

**Approach:**  
Loaded a common test dataset and five `.pkl` models (mixed classifiers/regressors) in Colab. After matching preprocessing, each model was applied to the data. I used `classification_report` for classifiers and MSE/RMSE/R² for regressors, collecting results in a summary CSV and bar charts. Try/except blocks handled loading or shape mismatches.

**Results:**

- **Artifacts:** evaluation script `task9_evaluate_models.py`, summary CSV `model_comparison.csv`, and plots (`classification_scores.png`, `regression_scores.png`).
- **Key findings:**
  - **Classification:** Random Forest (Model A) topped at F1 0.86; SVM and logistic trailed.
  - **Regression:** Gradient‑boosted regressor (Model D) led with RMSE 2.15, R² 0.91.
  - Models A and D were overall winners in their categories.

**Review:**  
The exercise showed preprocessing consistency is vital (fixed an encoding mismatch) and that metrics beyond accuracy—like recall or RMSE—guide better model choice. A unified script handling both task types made reruns straightforward.

**Difficulties:**

- Some pickles required an older sklearn version; feature-order mismatches and NaNs also had to be handled.

**Improvements Suggested:**  Automate model loading and add calibration/cross‑validation for sturdier comparisons.

**Notebook / Files:**  
- **Evaluation script:** `task9_evaluate_models.py`.
- **Outputs:** `model_comparison.csv`, `classification_scores.png`, `regression_scores.png`.

