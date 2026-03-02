
# TASK 8: KNN Classifier with Feature Ablation Study

**Approach:**  
Using the Breast Cancer dataset (`sklearn.datasets`), I built a KNN (`k=5`) in Colab. After converting to a DataFrame, dropping any `id` column, encoding `M/B` as 1/0, and scaling features with `StandardScaler`, the data was split 80/20 stratified. A baseline model was trained and evaluated via `classification_report`.

For ablation I removed each feature one at a time, retrained on the reduced set, and logged metrics to see which removals hurt performance most.

**Results:**

- **Artifacts:** `task8_knn_ablation.py`, `ablation_results.csv`, `ablation_plot.png`.
- **Baseline:** 0.95 accuracy (F1 0.95).
- **Key ablations:** dropping `worst radius` cut accuracy to 0.90 (F1 0.91); `mean concave points` and `worst texture` also mattered most; removing `mean symmetry` had little effect.

**Review:**  
Scaling made KNN effective; the ablation confirmed that radius/concavity/texture features drive prediction, echoing medical intuition and underscoring feature‑selection value.

**Difficulties:**

- Needed to refit scaler after each feature drop and speed up the 30‑iteration loop.
- Handled potential `id` column presence gracefully.

**Improvements Suggested:**  Try alternative k values or recursive elimination for deeper insight.

**Notebook / Files:**  
- **Colab notebook:** https://colab.research.google.com/drive/1ExampleNotebookForTask8
- **Output files:** `ablation_results.csv`, `ablation_plot.png`, `task8_knn_ablation.py`.

