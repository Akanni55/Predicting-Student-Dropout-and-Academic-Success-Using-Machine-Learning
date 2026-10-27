# Predicting Student Dropout and Academic Success Using Machine Learning

An exploratory Python machine learning project comparing support vector machines, random forests, and a stacked ensemble to classify student dropout outcomes.

**Portfolio focus:** data preparation, exploratory analysis, classification, hyperparameter tuning, and critical interpretation of model performance.

**Project status:** academic/portfolio analysis. The repository contains a notebook, its exported script, and a dataset; it does not contain a deployed prediction service or a validated student intervention system.

## Project objective

Student dropout can disrupt students' education and create challenges for institutional planning. This project explores whether student characteristics, financial indicators, and academic performance can distinguish dropout from non-dropout records, and compares the trade-offs between several classifiers.

The implemented task is **binary classification**, despite the broader repository title:

| Encoded target | Original outcomes | Meaning in this analysis |
| --- | --- | --- |
| `0` | Dropout | Recorded dropout |
| `1` | Enrolled + Graduate | Combined non-dropout class |

The code temporarily names the combined class `Still enrolled`. This README uses **non-dropout** because graduates are included. The model does not separately predict graduation or establish that currently enrolled students will eventually graduate.

A potential application is informing student-support research. No reduction in dropout, operational adoption, or causal impact has been measured.

## Repository structure

The project currently uses a flat structure:

| File | Purpose |
| --- | --- |
| [README.md](README.md) | Project overview, methodology, recorded results, and reproduction guidance |
| [Dropout Model implementations.ipynb](Dropout%20Model%20implementations.ipynb) | Main analysis, visualisations, model training, saved evaluation outputs, and subgroup diagnostics |
| [student dropout.csv](student%20dropout.csv) | Input dataset loaded by both implementations |

**Suggested review order:** read this README, inspect the notebook's preprocessing and model evaluation sections, then examine the script for the implementation details.

## Data overview

Inspection of the committed CSV gives:

| Property | Value |
| --- | --- |
| Records | 4,424 |
| Columns | 35: 34 input variables and `Target` |
| Original labels | Graduate: 2,209; Dropout: 1,421; Enrolled: 794 |
| Binary class balance | Dropout: 32.12%; non-dropout: 67.88% |
| Empty cells / exact duplicate rows | None found in the committed CSV |
| File format | Comma-separated CSV with a UTF-8 byte-order mark |

The source columns cover enrolment and demographic information, family background, financial status, curricular-unit activity and grades, and economic indicators. Absence of empty cells does not establish that every value or category is valid.

The code manually removes 25 columns, leaving these **nine modelling features**:

| Feature group | Retained columns |
| --- | --- |
| Student characteristics | `Displaced`, `Gender`, `Age at enrollment` |
| Financial indicators | `Debtor`, `Tuition fees up to date`, `Scholarship holder` |
| Academic performance | `Curricular units 1st sem (grade)`, `Curricular units 2nd sem (grade)` |
| Economic context | `GDP` |

The existing files do not document a verified upstream source, collection period, category codebook, or dataset licence. These should be confirmed before wider reuse; this README does not assign unverified meanings to numeric category codes or assume that public availability grants unrestricted reuse.

## Methodology

1. **Load and inspect:** read the CSV with pandas, inspect missing values, and review data types.
2. **Prepare the target and predictors:** remove the manually selected columns and combine Graduate/Enrolled into class `1`.
3. **Scale numerical features:** apply `StandardScaler` to age, both semester grades, and GDP. The remaining features retain their existing numeric codes.
4. **Explore the data:** generate descriptive statistics, histograms, box plots, categorical comparisons, a correlation heatmap, and target/gender distributions.
5. **Split the data:** use an 80/20 train/test split with `random_state=42`, producing 3,539 training records and 885 test records. The split does not explicitly stratify by target.
6. **Train and compare:** evaluate untuned and tuned versions of SVM, Random Forest, and a stacked ensemble.
7. **Examine subgroup performance:** generate classification reports for tuned SVM and Random Forest by gender, GDP, debtor status, and scholarship status.

### Models and tuning

| Model | Implementation | Search approach |
| --- | --- | --- |
| Support vector machine | `SVC` | 5-fold grid search over C, kernel, and gamma; 12 candidates |
| Random Forest | `RandomForestClassifier` | 5-fold grid search over tree count, depth, minimum split/leaf sizes, and bootstrap; 162 candidates |
| Stacked ensemble | Logistic Regression, SVC, and Random Forest base learners; Logistic Regression final estimator | 3-fold internal stacking; 3-fold grid search over 16 candidates using the training portion of a 10% sample |

All grid searches use `scoring='f1'`, which optimises **class 1 (non-dropout)** under the current encoding. The stacked model is subsequently fitted on the full training partition.

SMOTE and several other utilities are imported, but **no resampling is performed** in the implemented workflow. Logistic Regression is used within the ensemble; it is not reported as a standalone baseline.

## Model evaluation and recorded results

The workflow reports accuracy, classification reports, and confusion matrices. Tuned models also print binary precision, recall, and F1 explicitly.

The following values come from the notebook's **saved evaluation outputs**, not a fresh training run. Accuracy is rounded to four decimal places. Dropout recall is calculated from each saved confusion matrix as correctly identified dropouts divided by the 316 actual dropouts in the test set.

| Model | Accuracy | Precision: non-dropout | Recall: non-dropout | F1: non-dropout | Recall: dropout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stacked ensemble — untuned | 0.8350 | — | — | — | 0.6297 |
| Stacked ensemble — tuned | 0.8260 | 0.8158 | 0.9420 | 0.8744 | 0.6171 |
| SVM — untuned | 0.8282 | — | — | — | 0.6013 |
| SVM — tuned | 0.8282 | 0.8126 | 0.9525 | 0.8770 | 0.6044 |
| Random Forest — untuned | 0.8260 | — | — | — | 0.6551 |
| Random Forest — tuned | 0.8294 | 0.8266 | 0.9297 | 0.8751 | 0.6487 |

Dashes indicate that these metrics are available only at two-decimal precision in the untuned classification reports; they are omitted here to avoid implying equivalent reporting precision.

**Results consistency:** the final plotting section manually enters different accuracy/precision values for Random Forest and the ensemble. For example, the plot uses ensemble accuracy 0.8380, whereas the saved tuned evaluation reports 0.8260. This README uses the model evaluation outputs. The chart should be regenerated from calculated metrics before being reused.

### What the results mean

- **Among tuned models**, Random Forest has the highest recorded accuracy and non-dropout precision; SVM has the highest non-dropout recall and F1.
- **Across all six recorded runs**, the untuned ensemble has the highest accuracy. Tuning does not consistently improve the recorded test results.
- **Dropout detection remains the central weakness.** The tuned SVM identifies 191 of 316 dropouts and misses 125. Its 95.25% recall refers to non-dropout students, not students who dropped out.
- The tuned Random Forest identifies 205 of 316 dropouts, while incorrectly classifying 40 non-dropout students as dropout. This illustrates the trade-off relevant to student-support capacity.
- Always predicting non-dropout would achieve 569/885 = **64.29% accuracy on this test partition**. This is a calculated reference, not a separately trained baseline.
- These comparisons are exploratory. The validation limitations below prevent treating the results as unbiased deployment estimates or proof that one model is reliably superior.

## Limitations and interpretation

1. **Preprocessing leakage:** the scaler is fitted on the full dataset before splitting. Test-set information therefore influences preprocessing; scaling should be fitted within training-only pipelines and cross-validation folds.
2. **Stacking search contamination:** the 10% tuning sample is drawn from the full dataset before the final full-data split. Its tuning-training records can overlap the final test partition. Sample only from the training partition in a revised experiment.
3. **Prediction timing:** first- and second-semester grades are included. These results cannot be described as admission-time early warning. A deployment scenario would need an explicit prediction date and features available by that date.
4. **Objective mismatch and class imbalance:** default binary F1 favours evaluation of the combined non-dropout class. A dropout-support use case needs dropout-focused recall/precision, threshold analysis, and explicit costs for missed cases and false alerts.
5. **Unvalidated feature selection:** the 25 dropped columns are labelled “non-beneficial” in code without a documented selection study. That label does not establish that they lack predictive value.
6. **Age-group chart error:** age bands are constructed after age has been standardised, but the labels resemble ages in years. Those plots cannot support real-world age-band conclusions without rebuilding the groups from raw age.
7. **Limited fairness evidence:** subgroup classification reports are diagnostic only. They do not establish fairness, include uncertainty estimates, or demonstrate that sensitive attributes and proxies are suitable for operational decisions.
8. **Reproducibility and generalisation:** package versions are not pinned, some tuned estimators lack fixed random seeds, and evaluation uses one random holdout. There is no external or temporal validation, confidence interval analysis, probability calibration, or measured intervention outcome.
9. **Research implementation:** there is no packaged inference pipeline, saved model artefact, automated test suite, monitoring, or deployment configuration.

## Reproducing the analysis

### Local setup

Use Python 3 and Git. The original environment was Colab; its exact Python and dependency versions are not recorded. These commands provide a starting environment, not a version-locked recreation of the saved results.

```bash
git clone https://github.com/Akanni55/Predicting-Student-Dropout-and-Academic-Success-Using-Machine-Learning.git
cd Predicting-Student-Dropout-and-Academic-Success-Using-Machine-Learning
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`

Install the packages imported by the code and a notebook interface:

```bash
python -m pip install pandas numpy scipy matplotlib seaborn scikit-learn imbalanced-learn jupyterlab
python -m jupyterlab
```

Open `Dropout Model implementations.ipynb`, select the environment's Python kernel, restart it, and run all cells in order. Keep the CSV in the repository root and launch Jupyter from that directory because the data path is relative.

Alternatively, run the exported script from the repository root:

```bash
python SCRIPT
```

The script is Python despite having no extension. It prints results and opens plots; an interactive local session may require closing plot windows to continue. `imbalanced-learn` is needed by the existing SMOTE import even though SMOTE is not used.

### Colab option

Open the notebook in Google Colab, upload `student dropout.csv` to the runtime working directory, install any missing dependencies, and run the cells in order. The runtime upload is temporary.

### Expected outputs and checks

- Prepared dataset: 4,424 records, nine predictors, and one binary target.
- Split sizes: 3,539 training and 885 test records.
- Exploratory plots, grid-search logs, classification reports, confusion matrices, and subgroup reports.
- The Random Forest search alone requests 810 cross-validation fits, so a complete run can take time; grid searches use `n_jobs=-1`.
- Model scores may differ from the saved notebook because versions and some random seeds are unspecified.
- The final comparison chart remains hard-coded and will not automatically reflect a new run.

The README was checked against the committed code, saved outputs, and CSV. The full modelling workflow was not rerun for this documentation update.

## Recommended next iteration

- Reserve a stratified test set first; fit preprocessing and any sampling only inside training folds.
- Define the prediction point and evaluate dropout as the class of interest.
- Add a simple baseline, repeated or nested validation, and an untouched final evaluation.
- Rebuild age groups from raw ages and generate all result tables/charts directly from predictions.
- Document dataset provenance and coding, pin dependencies, and set estimator seeds.
- Validate subgroup performance and student-support usefulness before considering operational use.

These are proposed improvements, not implemented features.

## Portfolio summary

Developed a Python classification study using 4,424 student records, combining data preparation and exploratory analysis with SVM, Random Forest, and stacked-ensemble modelling. Compared tuned and untuned models through cross-validation searches, held-out classification reports, and subgroup diagnostics. The project demonstrates practical machine learning implementation and the importance of aligning evaluation metrics, data availability, and validation design with a student-support objective.
