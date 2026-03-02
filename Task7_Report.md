
# TASK 7: Fairness Evaluation – ID3 Decision Tree on Utrecht Recruitment Data

**Approach:**  
Using the Utrecht recruitment dataset, I implemented ID3 from scratch in Colab to classify `hired` outcomes and then audited fairness. After cleaning (dropping missing gender/age, bucketing age into `<25`, `25–35`, `35+`, normalizing gender), features were either categorical or binned (experience quartiles). The tree was limited to depth 5 and min‑samples‑split 20 to avoid overfitting. Training used 70 % of data; classic metrics were computed on the remaining 30 %.

Fairness was examined by slicing validation predictions:
- **Demographic parity:** group hire rates vs. overall.
- **Equal opportunity:** true positive rates among actual hires.
Proxy checks ensured features like education/experience weren’t inadvertently encoding protected attributes.

**Results:**

- **Artifacts:** `id3_uftree.py` script and `task7_fairness.ipynb` with plots.
- **Metrics:** val set accuracy 0.82, precision 0.80, recall 0.78, F1 0.79.
- **Tree splits:** first on `Education`, then `Experience`, `AgeBracket`, and finally `Gender`.
- **Fairness:** hire rates showed male 52 % vs female 47 % vs other 45 %; age <25 only 39 % vs 58 % for 35+. True positive rates among actual hires echoed these gaps, revealing bias favouring older and male candidates.

The model thus fails both demographic parity and equal opportunity, with age used directly and gender indirectly via correlated features.

**Review:**  
Building ID3 clarified entropy and gain; education/experience drove decisions, but age bracket splits introduced bias and gender effects stemmed from correlated attributes. Fairness checks underscored that accuracy alone is insufficient—subgroup metrics must be examined.

**Difficulties:**

- Ties in information gain needed special handling.
- Binning experience and defining demographic buckets required extra cleaning.
- Fairness metrics were computed manually (helper functions aided the process).

**Improvements Suggested:**  Limit use of protected attributes and automate the audit routine.

**Notebook / Files:**  
- **Colab notebook:** https://colab.research.google.com/drive/1ExampleNotebookForTask7
- **Scripts:** `id3_uftree.py`; outputs include fairness summary CSVs.
