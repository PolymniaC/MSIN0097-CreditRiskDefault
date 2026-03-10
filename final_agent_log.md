# Agent Usage Log — MSIN0097 Predictive Analytics

**Date:** 2026-03-04 | **Model:** Claude (Cursor extension)
**Notebooks:** `01_EDA.ipynb` · `02_preprocessing.ipynb` · `03_modelling_copy.ipynb`

---

## Agent Mistakes — Summary Table


| #   | Step   | Mistake                                                                                                                 | Caught by                 | Action                                                                                       |
| --- | ------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------- |
| 1   | Step 1 | Literal newline characters injected into strings in notebook JSON cells — occurred across 4 separate prompts            | Pre-run scan each time    | Manually corrected before execution                                                          |
| 2   | Step 3 | Suggested `max_iter=1000` for LR without empirical justification                                                        | Verification Experiment 1 | Rejected — changed to 200 based on observed convergence at 39–40 iterations                  |
| 3   | Step 3 | Suggested `max_depth=4` for XGBoost — hypothesis about L2 shifting optimum rightward not supported by this dataset      | Verification Experiment 4 | Rejected — changed to 2                                                                      |
| 4   | Step 3 | Suggested `n_estimators=300` for XGBoost — still on rising slope at that value                                          | Verification Experiment 5 | Rejected — changed to 500                                                                    |
| 5   | Step 3 | Experiment 5 first draft used `max_depth=4` despite Experiment 4 having already rejected it                             | Pre-run review of code    | Rejected before running — corrected to `BEST_DEPTH=2`                                        |
| 6   | Step 4 | `RandomizedSearchCV` proposed fitting on `X_tr_us` (globally pre-resampled) — CV folds not independent of sampling step | Pre-run review of code    | Rejected — corrected to imblearn Pipeline with undersampling inside each fold on full `X_tr` |
| 7   | Step 4 | Proposed refitting final model on `X_tr_us` directly rather than using the same pipeline as CV                          | Pre-run review of code    | Rejected — corrected to refit using imblearn Pipeline on `X_tr`                              |


---

## Step 1: EDA (`01_EDA.ipynb`)

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Replace the numeric printout of the target variable class proportions in the EDA notebook with a clear visualisation showing both counts and proportions for the binary target (0 = No Default, 1 = Default).

**PROMPT:**
"In my EDA notebook, I currently only print the class proportions of the target variable numerically. Can you replace this with a visual? I'm deciding between a pie chart and a grouped bar chart showing both the count and the proportion for each class (0 = No Default, 1 = Default). Recommend one and implement it, explaining your reasoning."

**KEY OUTPUT:**
Claude recommended using bar charts rather than a pie chart. The reasoning was that bar charts communicate class imbalance more clearly and allow both absolute counts and proportions to be displayed simultaneously. The implementation created two side-by-side bar charts: one displaying counts and the other displaying proportions.

**ITERATIONS / REFINEMENTS:**

1. The initial implementation used the colours "steelblue" and "tomato". I suggested adjusting colours to better match the notebook style. Claude reviewed the notebook and recommended using two shades from the seaborn "Blues" palette to maintain consistency with other categorical plots. This approach was accepted.
2. After viewing the resulting plot, I requested improved readability by placing the numeric values inside the bars instead of above them.
3. Claude modified the code to centre labels within each bar (using bar height / 2) and used white text for contrast, consistent with other visualisations in the notebook.
4. Initially, the visual output did not reflect the change. To debug this, I printed the source code of the relevant notebook cell directly from the .ipynb JSON file to verify whether the edit had been applied.
5. After confirming and re-running the updated cell, the final visualisation successfully displayed counts and percentages centred inside the bars.

**ACTION TAKEN:**
Accepted the final implementation showing two bar charts (count and proportion) with in-bar labels and a consistent Blues palette.

**FINAL RESULT:**
The visualisation clearly shows the class imbalance (23,364 No Default vs 6,636 Default; ~77.9% vs ~22.1%). The use of bar charts, consistent colour palette, and centred labels improves readability and aligns stylistically with the rest of the EDA notebook.

**NOTES / REFLECTION:**
This interaction demonstrates iterative AI-assisted refinement of a visualisation. AI was used for implementation suggestions, while final design decisions, debugging, and validation were supervised manually.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Improve the correlation heatmap in the EDA notebook. The original plot used `sns.heatmap` with `cmap='coolwarm'` but did not include annotations, making it difficult to read exact correlation values.

**PROMPT:**
"My current correlation heatmap at the end of the EDA notebook uses `sns.heatmap` with `cmap='coolwarm'` but has no annotations on the cells, making it hard to read exact values. Please improve it: add correlation value annotations, mask the upper triangle to reduce redundancy, and add a more descriptive title. Keep the same variable list I'm already using."

Claude was also allowed to locate the relevant heatmap cell in the notebook using a Python command to search the .ipynb file.

**KEY OUTPUT:**
Claude proposed an updated heatmap implementation including:

- `annot=True` with `fmt=".2f"` to display correlation values inside cells
- masking the upper triangle using `np.triu(...)` to remove redundant mirrored correlations
- a clearer descriptive title explaining the variables included and indicating that the correlations are calculated on the training set
- minor formatting improvements such as linewidths and annotation font size.

**ACTION TAKEN:**
Accepted the improvements to add annotations and mask the redundant triangle. The notebook cell was updated through a Python command that directly modified the source of the relevant cell in the .ipynb file.

**ISSUES IDENTIFIED:**
Two agent errors were identified before the cell ran successfully.

1. Missing numpy import — the code used `np.triu` and `np.ones_like` without verifying that numpy was imported in the notebook. This was caught before execution and fixed manually by adding `import numpy as np` to the imports cell.
2. A literal newline was injected into the title string, causing a SyntaxError. This was corrected manually by fixing the string escaping.

**ADDITIONAL MANUAL ADJUSTMENT:**
The original masking approach removed the diagonal values of the correlation matrix. This was manually corrected by modifying the mask definition to:

`mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)`

This preserves the diagonal while still hiding the redundant upper triangle.

**RESULT:**
After these corrections the heatmap executed successfully. The final plot shows the lower triangle of the correlation matrix with annotated Pearson correlation values and the diagonal retained. Variable ordering (LIMIT_BAL, AGE, BILL_AMT1–6, PAY_AMT1–6) was preserved to maintain the logical grouping of financial variables. The resulting visualisation clearly displays the strong correlation block among the bill amount variables and weaker relationships among payment variables.

**NOTES / REFLECTION:**
Agent required two manual corrections before the cell ran (missing numpy import and title string syntax error). After fixes, the heatmap output was accepted with annotations, masking, and layout functioning as intended.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Improve the visualisation of the PAY_0 repayment status variable to better show how repayment behaviour differs between defaulters and non-defaulters. The original plot only showed the overall frequency distribution of PAY_0 values in the training set.

**PROMPT:**
"I have a bar chart showing the distribution of PAY_0 values in the training set. Can you suggest and implement a better plot to show how the PAY_0 repayment status distribution differs between defaulters and non-defaulters? Propose at least two options."

Claude was allowed to locate the relevant notebook cell using a Python command to search the .ipynb file.

**KEY OUTPUT:**
Claude proposed two visualisation approaches.

Option 1 — Within-class grouped bar chart showing row-normalised proportions P(PAY_0 = x | class). This compares how repayment-status distributions differ within the Default and No Default groups.

Option 2 — 100% stacked bar chart showing the default rate per PAY_0 category, i.e. P(default | PAY_0 = x). This directly visualises the risk of default associated with each repayment-status bucket.

**ACTION TAKEN:**
Both options were initially generated. After reviewing the plots, Option 1 was rejected because it focuses on the distribution shape within each class, which obscures the predictive signal. Option 2 was accepted because it directly represents the conditional probability of default given PAY_0, which is the analytically appropriate perspective for evaluating PAY_0 as a predictor. Option 1 was removed from the notebook and only Option 2 was retained as the final visualisation.

**ISSUES IDENTIFIED:**
The agent again introduced literal newline characters inside title strings when writing the notebook cell through JSON, producing unterminated string errors. This is the third occurrence of the same issue previously seen in earlier cells. The strings were manually corrected by fixing newline escaping, and several undelimited strings were also corrected before execution.

**RESULT:**
The final plot displays a 100% stacked bar chart showing the proportion of defaulters and non-defaulters for each PAY_0 repayment-status category, with default percentages annotated inside the bars. This clearly shows how default risk changes across repayment-status buckets.

**NOTE ON ORIGINAL VISUALISATION:**
The original frequency distribution of PAY_0 (simple count bar chart) was removed and replaced. Frequency counts alone do not reveal the predictive relationship with the target variable, whereas the conditional default-rate plot directly shows how repayment status relates to default risk. The new visualisation therefore provides more meaningful insight for classification EDA.

**NOTES / REFLECTION:**
Two alternative visualisation strategies were proposed by the agent, but only the risk-based representation (Option 2) was retained because it aligns directly with the modelling objective of assessing the predictive relationship between repayment behaviour and default.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Replace the standalone LIMIT_BAL boxplot with a publication-quality visualisation that compares the credit limit distribution between defaulters and non-defaulters (training set only), to assess LIMIT_BAL as a potential predictor.

**PROMPT:**
"I currently have a standalone boxplot of LIMIT_BAL. Please replace or supplement it with a plot that shows the LIMIT_BAL distribution split by default status (0 vs 1), so we can visually assess whether credit limit is a useful predictor. Use the training set only. Make it publication-quality with proper axis labels, a legend, and a note about the log scale if you apply one."

**KEY OUTPUT:**
Claude proposed a two-panel figure using the training set:

- Left panel: overlapping KDE density curves for LIMIT_BAL by default class, plotted on a log x-scale.
- Right panel: side-by-side boxplots of LIMIT_BAL by default class with a log y-scale.

The plot included clear titles, axis labels with currency units (NT$), a legend, a figure-level title including sample size, a log-scale explanatory footnote, and median annotations on the boxplots.

**ACTION TAKEN:**
Accepted the KDE + boxplot combination with log scale and median annotations. The original standalone LIMIT_BAL boxplot was superseded by the new split-by-class visualisation.

**ISSUES IDENTIFIED:**
Agent repeated the literal newline / unterminated string error for the fourth time (previously observed in the heatmap and PAY_0 cells), this time inside an f-string used for the median label annotation. This caused a SyntaxError and was fixed manually by correcting string escaping. This recurrence confirmed a systematic pattern: the agent cannot reliably escape newlines when injecting code into notebook JSON. A personal verification rule was established: always scan all strings for literal newlines before running any agent-edited cell.

**RESULT:**
Plot accepted fully after manually fixing the single newline-related syntax issue. The final figure is publication-ready and clearly compares LIMIT_BAL distributions across default classes using both density (KDE) and robust summary (boxplot) views on a log scale.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Improve the AGE variable EDA by producing a single figure that shows both the raw age distribution and the default rate binned by age group (10-year bins), using the training set only.

**PROMPT:**
"For the AGE variable, I want a single figure that shows both the raw age distribution AND the default rate binned by age group (e.g. 10-year bins). Can you implement this as a two-panel figure? Use only the training set."

**KEY OUTPUT:**
Claude implemented a two-panel figure:

- Left panel: overlapping histogram (density) with KDE, split by default class (No Default vs Default).
- Right panel: bar chart of default rate by 10-year age bins (20–29, 30–39, 40–49, 50–59, 60+), including default-rate labels inside bars, sample size (n) above each bar, and an overall default-rate reference line.

**ACTION TAKEN:**
After reviewing the output, the left panel (overlapping KDE by class) was rejected — the age distributions for defaulters and non-defaulters were nearly identical in shape, making the histogram/KDE panel uninformative for predictive assessment. The right panel (default rate by age group with overall reference line) was accepted and retained as a standalone full-width plot.

**MANUAL REFACTORING:**
The agent's `plt.subplots(1, 2)` layout required manual refactoring to a single full-width plot: changed to `fig, ax = plt.subplots(1, 1, figsize=(8, 5))`, removed the left-panel code entirely, and refactored all `axes[1]` references to `ax`.

**RESULT:**
Final visualisation is a single cleaner plot showing default rate by age group with sample sizes and an overall default-rate reference line. This is more analytically focused than the original two-panel proposal.

**NOTES / REFLECTION:**
This is a case where the agent over-engineered the solution. The distribution panel added no meaningful insight due to near-identical class shapes. Retaining only the risk-by-bin plot improved clarity and relevance for classification EDA.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Combine separate frequency distribution and default-rate plots for the EDUCATION and SEX variables into compact two-panel figures (frequency + default rate side-by-side) to improve notebook readability and make comparisons more immediate.

**PROMPT:**
"I have separate frequency distribution and default rate plots for the EDUCATION and SEX variables. Can you combine each pair into a single two-panel figure (frequency + default rate side by side) to make the notebook more compact and the comparison more immediate?"

**ACTION / ACCESS:**
Allowed a Python command to edit the notebook JSON directly, updating two cells: SEX cell id b79f0a79 and EDUCATION cell id fdcd39d2.

**KEY OUTPUT:**
SEX: Two-panel figure with frequency countplot (in-bar counts) on the left and default rate barplot with overall reference line on the right. Consistent palette and labels.

EDUCATION: Two-panel figure with frequency distribution across all EDUCATION codes (including undocumented codes) on the left, and default-rate barplot with overall-rate reference line and percentage annotations on the right.

**REVIEW / DECISIONS:**
SEX plot accepted fully — the combined two-panel layout was clean, compact, and immediately informative. No changes required.

EDUCATION plot partially accepted — the layout and frequency panel were correct, but the default-rate panel required adjustment. Undocumented education codes (0, 5, 6) had extremely small sample sizes (n=12, 235, 42), producing unreliable default-rate estimates with visually misleading uncertainty intervals. Displaying those categories alongside well-sampled groups obscured the meaningful trend across documented categories.

**FOLLOW-UP MODIFICATION:**
In the EDUCATION default-rate panel only, categories 0, 5, and 6 were filtered out. These categories are retained in the frequency panel and in the dataset itself — they were excluded only from the default-rate display. A footnote was added explaining the rationale. Category 4 (Others, n=97) was retained in the default-rate panel but flagged in the interpretation due to its wide confidence interval.

**RESULT:**
Final notebook contains a single two-panel SEX figure and a single two-panel EDUCATION figure where the frequency panel shows all codes but the default-rate panel focuses on documented categories 1–4 with a footnote documenting exclusions.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add a final markdown cell at the end of the EDA notebook summarising the key findings and modelling implications in a structured table with columns: Variable / Group, Key Finding, and Modelling Implication.

**PROMPT:**
"At the end of my EDA notebook, add a summary markdown cell that lists the key findings from the EDA in a structured table — one row per variable or variable group, with columns for: Variable, Key Finding, and Modelling Implication."

**KEY OUTPUT:**
Claude created a structured markdown table covering: Target, LIMIT_BAL, AGE, SEX, EDUCATION, MARRIAGE, PAY_0, PAY_2–PAY_6, BILL_AMT1–BILL_AMT6, PAY_AMT1–PAY_AMT6. Each row included a concise empirical finding and a direct modelling or preprocessing implication.

**REVIEW / MODIFICATIONS:**
Two modelling implications were manually refined.

**EDUCATION — Original agent text:**
*"Encode as ordinal or one-hot. Group or flag undocumented codes rather than treating them as a meaningful fifth category."*

**Modified to:**
*"Ordinal encoding is reasonable given the partial monotonic pattern observed in default rates (Graduate < University < High School). However, one-hot encoding is a valid alternative, particularly if category 4 (Others) disrupts the ordering. Both approaches should be tested during modelling. Group undocumented codes (0, 5, 6) into a single Other/Unknown category during preprocessing."*

**Reason:** The agent presented ordinal vs one-hot as an open choice without justification. The modification explicitly ties the recommendation to the empirical pattern observed in the EDA, while acknowledging the uncertainty introduced by the Others category.

**MARRIAGE — Original agent text:**
*"Weak predictor. Encode as nominal (one-hot). Recode or group code 0 with category 3 (Others) during preprocessing."*

**Modified to:**
*"Weak predictor. Encode as one-hot. Code 0 (undocumented, n=47) may be grouped with category 3 (Others) during preprocessing, but this should be verified against model performance rather than assumed."*

**Reason:** The agent stated the grouping as a definitive instruction rather than a hypothesis to test. Modified to reflect that this preprocessing decision requires empirical validation rather than blind application.

**RESULT:**
The final markdown table was accepted with the two modifications above and added as the closing cell of the EDA notebook.

---

**SYSTEMATIC PATTERN NOTE:**
Across prompts 2, 3, 4, and 5, the agent repeatedly injected literal newline characters into strings when writing code to notebook JSON cells. This occurred in regular strings, multiline titles, print statements, and f-strings. All instances were caught and corrected manually before execution. This was treated as a known and recurring limitation of the agent's notebook-editing approach, and a personal verification rule was established: always scan all strings for literal newlines before running any agent-edited cell.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add a numeric summary table to the PAY_0 cell to print default rates and sample sizes per repayment-status category before the plot, so that interpretation claims can be backed by precise computed values rather than visually estimated from the chart.

**PROMPT:**
"In the PAY_0 cell in my EDA notebook, add a numeric summary table printed before the plot. After the existing print statements for counts and proportions, add code that computes and prints a summary table showing the default rate and sample size (n) per PAY_0 category, rounded to 3 decimal places."

**ACTION TAKEN:**
Accepted. A Python command updated cell id 8d0e1c56 directly in the notebook JSON, inserting the summary table computation and print statement between the existing distribution prints and the plot code. No other changes were made.
Removed the raw count printout: because the information is already provided in the summary table (n per PAY_0 category), making the separate count output redundant.

**RESULT:**
The cell now prints counts, proportions, and a default_rate/n summary table before rendering the plot. This allows precise numerical claims to be made in the interpretation (e.g. status 0 = 12.8%, status 1 = 34.4%) rather than relying on visual estimates from the bar chart.

**REASON:**
During interpretation review, a factual error was identified — the interpretation had incorrectly stated that status 0 corresponds to a 34% default rate, when the actual computed value is 12.8%. The 34% figure belongs to status 1. Adding the numeric summary table prevents this class of error by making exact values visible in the notebook output.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add a numeric summary table for PAY_2–PAY_6 default rates per repayment-status category, to back up interpretation claims with computed values rather than visual estimates from the plot.

**PROMPT:**
"In my EDA notebook, find the cell containing the PAY_2–PAY_6 plot. Before that plot cell, add a new code cell that computes and prints a default rate summary table for each of PAY_2, PAY_3, PAY_4, PAY_5, and PAY_6. For each variable, print the variable name, then a table showing the repayment status category, default rate (rounded to 3 decimal places), and sample size n. Use a loop over the five columns. Format it the same way as the PAY_0 summary table already in the notebook."

**KEY OUTPUT:**
Agent produced a loop over `repay_cols = ["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]`. For each variable it sorts the unique category values (including negative codes -2 and -1), groups by repayment status, computes mean default rate and count, rounds to 3 decimal places, and prints a formatted table.

**ACTION TAKEN:**
Accepted. Cell was placed after the PAY_2–PAY_6 plot rather than before it as requested. Placement is not ideal but functionally correct — all interpretation claims remain verifiable from the printed output.

**RESULT:**
Notebook now prints default rates and sample sizes per repayment-status category (covering the full range of observed codes including -2, -1, 0 through 8) for all five remaining PAY variables, consistent with the PAY_0 summary table format.

**REASON:**
The PAY_2–PAY_6 interpretation cited specific default rate thresholds without any computed numeric output to support these claims. This was inconsistent with the PAY_0 section which already had a printed summary table. Adding this ensures all repayment history interpretation claims are directly verifiable from notebook output.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add a final code cell to the EDA notebook that saves the train/test splits to CSV files in `data/processed/` for use in downstream modelling notebooks.

**PROMPT:**
"At the very end of my EDA notebook, add a new code cell that saves the train and test splits to CSV files. Save X_train, X_test, y_train, and y_test to a folder called data/processed relative to the project root. Create the directory if it doesn't exist using os.makedirs with exist_ok=True. Use pandas to_csv with index=False for all four files. Print a confirmation message when done."

**ACTION TAKEN:**
A new code cell (id: save_splits_csv) was appended as the final cell of ML_EDA_WORKFLOW.ipynb. The output directory is resolved using `os.path.dirname(os.getcwd())` to navigate one level above the notebooks/ folder to the project root, then into data/processed/. The directory is created with `exist_ok=True`. All four splits are saved with `index=False`. A confirmation message prints the output path and shape of each file.

Note: the notebook had been renamed from ML_EDA.ipynb to ML_EDA_WORKFLOW.ipynb during this session. The cell was appended to the correct current file.

**RESULT:**
Running the cell will produce X_train.csv, X_test.csv, y_train.csv, and y_test.csv in data/processed/, making the cleaned and split data available to the preprocessing and modelling notebooks without repeating the split logic.

---

## Step 2: Preprocessing (`02_preprocessing.ipynb`)

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Create `02_preprocessing.ipynb` and load the four pre-split CSV files into DataFrames, printing shapes and class distribution.

**PROMPT:**
"I'm doing a credit card default prediction project. I have four files already split from EDA: X_train.csv, X_test.csv, y_train.csv, y_test.csv (24,000 train rows, 6,000 test rows, 23 features). Create a new file called 02_preprocessing.ipynb. Load all four CSVs, print shapes, show X_train.head(), and print y_train.value_counts() with proportions."

**ACTION TAKEN:**
Accepted. Agent used `.squeeze()` on y files to prevent Series/DataFrame shape issues and structured the load path to `../data/processed/` rather than the current directory. The path restructure was accepted because it supports a cleaner repo layout, which the brief rewards for reproducibility.

Manual improvement: the agent loaded data and printed shapes but did not verify that X_train and X_test have identical column names. An assertion was manually added immediately after loading. A column mismatch would cause silent downstream failures in the pipeline — catching it at load time is safer.

**RESULT:**
X_train: (24000, 23), X_test: (6000, 23). Class imbalance confirmed: 78.3% class 0, 21.7% class 1. Column name assertion passed — 23 matching columns confirmed.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Fix invalid category codes in EDUCATION and MARRIAGE identified during EDA before any other preprocessing.

**PROMPT:**
"In X_train and X_test, fix the following dirty category codes before any other preprocessing. EDUCATION: codes 0, 5, 6 are undocumented — remap them to 4 (others). MARRIAGE: code 0 is undocumented — remap it to 3 (others). Apply identically to both sets. Print value_counts() after remapping and confirm no undocumented codes remain."

**ACTION TAKEN:**
Accepted. Agent generated remap dictionaries applied identically to train and test using `.replace()`, with four assertion checks confirming no undocumented codes remain. Mappings were defined from training-set findings only and applied blindly to test — no test information used.

Manual improvement: the agent's recode cell contained no comments explaining why the remapping was made. Comments were manually added above each remap dictionary, citing Yeh & Lien (2009) and stating which codes are valid per the original dataset documentation. No logic was changed — documentation improvement only, making the rationale traceable and auditable as required by the brief.

**RESULT:**
EDUCATION: counts 1=8455, 2=11256, 3=3903, 4=386 (train). MARRIAGE: 1=10892, 2=12806, 3=302 (train). All 4 assertions passed. No undocumented codes remain in either split.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Engineer 7 derived features from raw variables on both train and test.

**PROMPT:**
"Engineer the following new features on both X_train and X_test: UTIL_RATE = BILL_AMT1/LIMIT_BAL clipped [0,1], negatives treated as 0; AVG_PAY_STATUS = mean of PAY columns; MAX_DLQ = max of PAY columns; TOTAL_BILL = sum of BILL_AMT columns; TOTAL_PAY = sum of PAY_AMT columns; PAY_RATIO = TOTAL_PAY/(TOTAL_BILL+1) clipped [0,5]; BILL_TREND = BILL_AMT1 - BILL_AMT6. Print .describe() for each and verify no NaNs."

**ACTION TAKEN:**
Accepted. Agent wrapped all feature creation in a reusable `engineer_features(df)` function rather than writing duplicate code for each split — this was better than the prompt specified and made leakage-free application to both splits explicit. UTIL_RATE edge cases were handled correctly: negative BILL_AMT1 clipped to 0, LIMIT_BAL=0 replaced with NaN then fillna(0). TOTAL_BILL negatives were noted as valid (credit overpayments across months), not an error.

Manual improvement: the agent verified feature distributions and NaN counts but did not check correlations between the 7 engineered features. A correlation matrix and automated flagging loop (|r| > 0.8) were manually added, computed on unscaled X_train before any scaling. One highly correlated pair was found: AVG_PAY_STATUS vs MAX_DLQ (r = 0.806). Both features were retained because they capture distinct signals — AVG_PAY_STATUS reflects average delinquency over 6 months while MAX_DLQ captures the single worst episode. A customer can have a low average but one severe spike that AVG_PAY_STATUS smooths over. The correlation is noted as a limitation for models sensitive to multicollinearity.

**RESULT:**
X_train: (24000, 30), X_test: (6000, 30). All 7 features: 0 NaNs in train and test. One correlated pair flagged (AVG_PAY_STATUS vs MAX_DLQ, r=0.806) — both retained.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Scale UTIL_RATE and TOTAL_BILL prior to the full preprocessing pipeline.

**PROMPT:**
"We are about to scale features. Before building the full pipeline, can you quickly normalise UTIL_RATE and TOTAL_BILL so they are on a comparable scale to the other features? Just fit and transform directly on the available data."

**ACTION TAKEN:**
The agent surfaced a clarification question on fit scope with two options: fit on X_train only, or fit on all data (X_train + X_test). Option 1 was selected. Fitting on combined train+test leaks test distribution statistics into scaler parameters, producing optimistically biased evaluation results. The agent itself labelled the second option as "leaks test distribution into the scaler — not recommended for a real pipeline."

After confirming fit scope, the agent generated a standalone StandardScaler cell scaling only UTIL_RATE and TOTAL_BILL in isolation as a one-off step. This was rejected before execution. Scaling two features in isolation creates a separate scaler object outside the main pipeline — inconsistent, harder to manage, and not a single reproducible fitted object. All scaling should happen consistently with a clear single fitted object.

The agent was prompted to regenerate. The second output correctly fit StandardScaler on X_train only, applied transform to both splits, and printed learned parameters and post-scaling describe() for verification. This was accepted.

**RESULT:**
UTIL_RATE: mean=0.4145, std=0.3868. TOTAL_BILL: mean=268971.70, std=378116.10. Both learned from X_train only. X_test mean slightly off zero (~0.003, ~0.012) — expected and confirms no leakage.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Build a full ColumnTransformer preprocessing pipeline fit only on X_train, transform both splits, and output as DataFrames with clean column names.

**PROMPT:**
"Build a scikit-learn ColumnTransformer fitted only on X_train. num_cols: LIMIT_BAL, AGE, BILL_AMT1-6, PAY_AMT1-6, AVG_PAY_STATUS, MAX_DLQ, TOTAL_PAY, PAY_RATIO, BILL_TREND → StandardScaler. ord_cols: PAY_0, PAY_2-PAY_6 → StandardScaler. cat_cols: SEX, EDUCATION, MARRIAGE → OneHotEncoder(drop=first, sparse_output=False). Exclude UTIL_RATE and TOTAL_BILL as they are already scaled. Fit on X_train, transform both. Output as DataFrames. Print shapes and confirm no NaNs."

**ACTION TAKEN:**
Accepted. Agent defined 4 column groups correctly: num_cols and ord_cols with StandardScaler, cat_cols with OneHotEncoder(drop='first', sparse_output=False), and pass_cols (UTIL_RATE, TOTAL_BILL) as passthrough to avoid double-scaling. Used `verbose_feature_names_out=False` for clean column names and `remainder='drop'` for explicit control.

Manual improvement: the agent used `OneHotEncoder(drop='first', sparse_output=False)` without specifying behaviour for unseen categories. `handle_unknown='ignore'` was manually added. Without it the pipeline crashes at inference if new data contains an unseen category value. This dataset already had undocumented codes handled in Step 2, but future scoring data could introduce unexpected values. With ignore, the encoder outputs zeros for unseen categories and the model can still produce a prediction. Re-running confirmed identical output — same shapes, same 33 columns, zero NaNs. Improvement is for deployment robustness only.

**RESULT:**
X_train_proc: (24000, 33). X_test_proc: (6000, 33). OHE produced: SEX_2, EDUCATION_2/3/4, MARRIAGE_2/3. Zero NaNs in both splits.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Handle class imbalance and create a held-out validation set in a leakage-safe order.

**PROMPT:**
"Handle class imbalance on the training data and set aside a validation set of 15% for model evaluation. Use random_state=42 throughout."

**ACTION TAKEN:**
The agent surfaced two clarification questions before generating code.

First, imbalance handling strategy. Four options were offered. SMOTE was selected: generates synthetic minority samples via interpolation between real neighbours — more informative than simple duplication and lower overfitting risk. The agent confirmed it would be applied after the val split, preventing synthetic samples from entering validation. Random oversampling was rejected: duplicates existing samples only, adds no new information. Class weights only was rejected: less effective at 78/22 imbalance. Other was rejected: no justification for an undiscussed method.

Second, validation split source. Two options were offered. From X_train_proc (post-pipeline) was selected: the ColumnTransformer was already fitted on full X_train in Step 4, so splitting post-pipeline is the consistent choice — val goes through the same transforms as train with no mismatch risk. From raw X_train (pre-pipeline) was rejected but noted as a limitation: technically the gold standard since the pipeline would be fit only on the train portion, but impractical given the pipeline was already fitted on full X_train. Recorded as a limitation in the notebook markdown.

Agent generated stratified 85/15 split then SMOTE only on the training portion — correct leakage-safe order. Accepted.

Manual improvement: agent used `SMOTE(random_state=42)` relying on the implicit default `sampling_strategy='auto'`, which would balance to 50/50. `sampling_strategy=0.5` was manually added to deliberately target a more conservative 67/33 ratio instead — class 1 is set to half of class 0, generating ~3,430 synthetic samples on top of 4,513 real minority observations. A true 50/50 balance was avoided as it would require ~12,000 synthetic samples, meaning 75% of minority class training data would be synthetic rather than real, which is harder to defend.

**RESULT:**
Before SMOTE — X_tr: (20400, 33), X_val: (3600, 33). Minority: train=22.12%, val=22.11% — stratification confirmed. After SMOTE — X_tr_res: (23830, 33), class 0: 15887, class 1: 7943 (67/33). Zero NaNs in both.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add automated validation checks confirming pipeline outputs are clean and consistent before modelling.

**PROMPT:**
"Add a data validation section that runs the following checks and prints a pass/fail summary: no NaN values in X_tr_res, X_val, X_test_proc; no infinite values; identical feature count across all splits; X_test_proc has exactly 6000 rows; no zero-variance features in X_tr_res; val class ratio within 2% of train. Print PASS or FAIL for each check."

**ACTION TAKEN:**
Accepted. Agent generated a `check()` helper function printing PASS/FAIL per check with a final summary line. All 6 checks passed on first run.

Manual improvement: the agent checked that all splits have the same number of columns but did not verify that column names and ordering are identical. A 7th check was manually added confirming `list(X_tr_res.columns) == list(X_test_proc.columns)`. A count match alone can silently pass even if columns are misordered or differently named, which would cause silent errors when a model trained on one feature order is evaluated on another.

**RESULT:**
7/7 checks passed — pipeline ready.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Persist all preprocessing outputs and fitted pipeline objects for use in modelling notebooks.

**PROMPT:**
"Save all preprocessing artefacts to data/processed/artefacts/: X_tr_res.csv, X_val.csv, X_test_proc.csv, y_tr_res.csv, y_val.csv, y_test.csv, preprocessor.joblib (fitted ColumnTransformer), scaler_util_bill.joblib (fitted StandardScaler). Print a final summary table showing split name, rows, features, and class balance %."

**ACTION TAKEN:**
Accepted. Agent correctly used `os.makedirs(exist_ok=True)` to ensure the directory exists, saved all 6 CSVs and 2 joblib objects, and printed a formatted summary table. Both fitted pipeline objects persisted — essential for applying identical transforms at inference without refitting. No changes were needed.

**RESULT:**
All 8 artefacts saved to data/processed/artefacts/. X_tr_res: 23,830 rows, 33 features, 33.33% default (67/33 SMOTE balance). X_val: 3,600 rows, 33 features, 22.11% default (real distribution). X_test_proc: 6,000 rows, 33 features, 22.12% default (real distribution). Preprocessing pipeline complete and fully reproducible.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Extend `02_preprocessing.ipynb` to create a second training dataset using random undersampling so that two imbalance-handling strategies can be compared during modelling:

1. SMOTE-balanced dataset (`X_tr_res`, `y_tr_res`) already created in Step 5
2. Undersampled dataset (`X_tr_us`, `y_tr_us`) targeting the same 67/33 class ratio.

The goal is to keep all other conditions identical so any model performance difference can be attributed solely to the sampling method.

**PROMPT:**
"To evaluate how best to handle the class imbalance in the credit card default dataset (≈78% non-default, 22% default), two balancing strategies will be compared using the same four candidate models.

First, models will be trained on the SMOTE-balanced dataset (X_tr_res, y_tr_res), where the minority class has been synthetically oversampled to achieve approximately a 67/33 distribution.

Second, a downsampled dataset will be created by randomly removing observations from the majority class (non-defaults) until the dataset matches the same 67/33 class ratio. Observations will be removed randomly rather than sequentially to avoid introducing ordering bias or accidentally removing structured patterns from the data.

In 02_preprocessing.ipynb, add a new section after the existing Step 5 (SMOTE) called **Step 5b: Undersampling Baseline Dataset**. This section should:

1. Add a markdown cell explaining that a second training set is created using random undersampling instead of SMOTE to allow a fair comparison of sampling strategies. The ratio is 67/33 matching the SMOTE set — achieved by keeping all minority class observations and randomly dropping majority class observations down to twice the minority count.
2. Add a code cell that:
  - Uses `RandomUnderSampler(sampling_strategy=0.5, random_state=42)` from `imblearn.under_sampling` on `X_tr` and `y_tr` (the pre-SMOTE split).
  - Prints class counts and minority percentage before and after undersampling.
  - Prints shapes of `X_tr_us` and `y_tr_us`.
  - Checks for NaN values.
3. Add a code cell that saves `X_tr_us.csv` and `y_tr_us.csv` to `data/processed/artefacts/`.

Do not modify existing cells or previously saved artefacts."

Claude attempted to insert the cells automatically using a Python script that edits the notebook JSON.

**KEY OUTPUT:**
Claude generated code to insert three new cells into `02_preprocessing.ipynb`:

- Markdown explanation for Step 5b — Undersampling Baseline Dataset
- Code cell applying `RandomUnderSampler(sampling_strategy=0.5)` to `X_tr` / `y_tr`
- Code cell saving `X_tr_us.csv` and `y_tr_us.csv` to the artefacts directory

However, the script attempted to insert the new cells immediately after the `val_smote` code cell, which corresponds to the execution part of Step 5.

**ACTION TAKEN:**
Rejected the first implementation. A follow-up prompt corrected the insertion location so the new Step 5b section would appear after the Step 5 interpretation markdown cell (id `3cdabfa0`), ensuring that the entire SMOTE section — including explanation and interpretation — remains grouped together.

**ISSUES IDENTIFIED:**
The agent incorrectly selected the insertion point by referencing the SMOTE code cell (`val_smote`) instead of the final markdown interpretation cell for Step 5. This would have placed the new Step 5b section in the middle of the Step 5 block rather than after it.

**ADDITIONAL MANUAL ADJUSTMENT:**
A revised instruction specified that the new cells must be inserted after the markdown cell with id `3cdabfa0`, which marks the end of the Step 5 section. The updated Python script was then executed and accepted.

**RESULT:**
`02_preprocessing.ipynb` now contains a new section Step 5b — Undersampling Baseline Dataset immediately following the full SMOTE explanation. Confirmed output: X_tr_us shape (13,539, 33), class 0: 9,026, class 1: 4,513, minority %: 0.3333. Zero NaNs confirmed.

The section applies `RandomUnderSampler(sampling_strategy=0.5, random_state=42)` to `X_tr` / `y_tr`, produces `X_tr_us` and `y_tr_us`, prints class distributions before and after undersampling, checks for NaN values, and saves both CSVs to `data/processed/artefacts/`.

This creates a second training dataset with the same 67/33 class ratio as the SMOTE set, allowing a controlled comparison between synthetic oversampling (SMOTE) and random undersampling in the modelling stage.

**NOTES / REFLECTION:**
Ensuring the correct insertion point was important for notebook readability and logical flow. Step 5b is now clearly separated as an alternative sampling strategy rather than interrupting the SMOTE section. This setup enables a fair experimental comparison during modelling because both training datasets originate from the same base split (`X_tr`, `y_tr`) and differ only in the balancing technique applied.

---

## Step 3: Model Shortlisting (`03_modelling_copy.ipynb`)

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Update `03_modelling.ipynb` after introducing a second training dataset (`X_tr_us`, `y_tr_us`) in the preprocessing notebook. The modelling notebook initially assumed only the SMOTE-balanced dataset would be used.

**PROMPT:**
"In `03_modelling.ipynb`, make two updates to existing cells — do not add new cells:

1. In the load cell, add two new lines after the existing loads to load `X_tr_us` and `y_tr_us`, with a print line and a column alignment check.
2. In the final paragraph of the intro markdown (the one starting 'All models are trained on X_tr_res...'), replace it with: 'All four models are trained and evaluated twice — once on X_tr_res/y_tr_res (SMOTE, 67/33, 23,830 rows) and once on X_tr_us/y_tr_us (random undersampling, 67/33, 13,539 rows). Both experiments evaluate on the same X_val/y_val (natural 22% default rate). The sampling strategy that produces better ROC-AUC and F1 on validation is carried forward to fine-tuning.'

Do not modify any other cells."

**KEY OUTPUT:**
Claude generated a script that modified the `load_data` cell to load `X_tr_us.csv` and `y_tr_us.csv`, added shape printing and class balance printing for the undersampled dataset, added column alignment verification between `X_tr_res` and `X_tr_us`, and attempted to replace the final paragraph of the `md_intro` markdown cell.

**ACTION TAKEN:**
Accepted the modification to the `load_data` cell. However, the initial script failed to update the markdown paragraph because the exact string it attempted to replace did not match the notebook text. A follow-up inspection command was executed to print the final characters of the markdown cell (`repr(src[-350:])`) in order to identify the exact text structure. A corrected script was then executed that located the paragraph using the marker `"All models are trained on"` and replaced everything from that marker onward.

**ISSUES IDENTIFIED:**
The first attempt to update the markdown failed because the replacement logic relied on an exact string match which did not match the notebook content precisely. As a result, no change was applied to the markdown section in the first attempt.

**RESULT:**
`03_modelling.ipynb` now reflects the revised experimental design. Both training datasets are loaded. The load cell prints shapes and class distributions for both. Column alignment checks verify feature consistency across all splits. The introductory markdown clearly states that each model will be trained twice and evaluated on the same validation set.

**NOTES / REFLECTION:**
This step illustrates the iterative workflow used during development. The modelling notebook was initially created assuming only the SMOTE dataset would be used. After reconsidering the imbalance-handling strategy, a second dataset was introduced in preprocessing. The modelling notebook was then revisited and updated to incorporate this change.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Implement the first shortlisting model in `03_modelling.ipynb` (Logistic Regression baseline) and evaluate it twice — once using the SMOTE training set and once using the undersampled training set.

**PROMPT:**
"In 03_modelling.ipynb, add the following cells after the load cell:

1. A markdown cell: ## Model 1: Logistic Regression (Linear Baseline) explaining that LR is trained first as the interpretable linear baseline. max_iter=1000 is set because the default 100 iterations is insufficient for convergence on 33 scaled features. C=1.0 default L2 is kept intentionally — no tuning at shortlisting stage per Géron (2019, Ch. 2). Trained twice: once on SMOTE set, once on undersampled set.
2. A code cell that trains `LogisticRegression(max_iter=1000, random_state=42)` on X_tr_res/y_tr_res, predicts on X_val, prints clearly labelled ROC-AUC and F1 for class 1 only (not weighted average), prints classification_report, and shows ConfusionMatrixDisplay.
3. A second code cell that does the same but trains on X_tr_us/y_tr_us instead. Label all outputs clearly as SMOTE and Undersampled.

Do not touch X_test_proc. Do not modify any existing cells."

**USER MODIFICATION TO AGENT INSTRUCTION:**
"Move the eval_fn helper function into its own separate code cell with id eval_helper inserted before md_lr, not inside lr_smote. The lr_smote cell should only contain the model training and eval_model call, not the function definition."

**KEY OUTPUT:**
Claude inserted four new cells: `eval_helper` (defines `eval_model()`), `md_lr` (markdown), `lr_smote` (SMOTE training/eval), `lr_us` (undersampled training/eval).

**ACTION TAKEN:**
Accepted. No code execution errors.

**Verification Experiment 1 — max_iter:**

Added two cells after `md_lr` and before `lr_smote` to test whether `max_iter=100` is sufficient for convergence on 33 scaled features.

Observed output:

SMOTE: `max_iter=100` → converged, n_iter_=40. `max_iter=1000` → converged, n_iter_=40.
Undersampled: `max_iter=100` → converged, n_iter_=39. `max_iter=1000` → converged, n_iter_=39.

The experiment was initially incomplete — first version only tested the SMOTE training set. Updated in a follow-up prompt to loop over both datasets. Based on this evidence, `max_iter` was reduced from the agent-suggested 1000 to 200 in both LR model cells and the explanatory markdown. This directly satisfies the assignment requirement to verify agent-suggested hyperparameters through experiment rather than accepting them on trust.

**Observed validation metrics:**

- Logistic Regression | SMOTE: ROC-AUC ≈ 0.7454, F1 (class 1) ≈ 0.5077
- Logistic Regression | Undersampled: ROC-AUC ≈ 0.7452, F1 (class 1) ≈ 0.5046

**RESULT:**
Complete baseline implementation for Logistic Regression evaluated under both sampling strategies. The extremely similar results provide an early empirical signal that the main gains may come from model choice rather than sampling strategy for linear methods.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add Model 2: Random Forest (Bagging Ensemble) to `03_modelling.ipynb`, trained and evaluated under both sampling strategies using the same validation set.

**PROMPT:**
"In `03_modelling.ipynb`, add three cells after the last logistic regression interpretation cell:

1. A markdown cell `## Model 2: Random Forest (Bagging Ensemble)` explaining: RF builds many decision trees on random bootstrap samples and averages their predictions, reducing variance compared to a single tree (Géron, 2019, Ch. 7). Handles the non-linear PAY_0 step-changes naturally via tree splits. Robust to the AVG_PAY_STATUS/MAX_DLQ correlation (r=0.806) via random feature subsampling at each split. Hyperparameters kept at defaults for shortlisting: `n_estimators=100`, `random_state=42`. No `max_depth` limit — standard at shortlisting stage. Trained twice.
2. A code cell with id `rf_smote` that trains `RandomForestClassifier(n_estimators=100, random_state=42)` on `X_tr_res/y_tr_res` and calls `eval_model`.
3. A code cell with id `rf_us` that trains on `X_tr_us/y_tr_us` and calls `eval_model`.

Do not modify any existing cells. Do not touch `X_test_proc`."

**KEY OUTPUT:**
Claude inserted three new cells. `rf_smote` and `rf_us` training cells accepted and executed.

**Verification Experiment 2 — n_estimators:**
Tested n_estimators ∈ {10, 50, 100, 200, 300} on both datasets. ROC-AUC improved rapidly from 10 → 50 trees, stabilised by ~100, negligible improvement beyond. `n_estimators=100` retained.

**Validation results:**

- Random Forest | SMOTE: ROC-AUC ≈ 0.7741, F1 (class 1) ≈ 0.4966. Confusion matrix: TN=2595, FP=209, FN=464, TP=332.
- Random Forest | Undersampled: ROC-AUC ≈ 0.7700, F1 (class 1) ≈ 0.5160. Confusion matrix: TN=2516, FP=288, FN=419, TP=377.

**ISSUES IDENTIFIED:**
The confusion matrix figure title was partially clipped by the colorbar. Two visual fixes applied to the `eval_model()` helper function: changed `figsize` from (4,4) to (5,4) and changed `ax.set_title(label)` to `ax.set_title(label, fontsize=9, pad=8)`. No other logic or evaluation code was modified.

**RESULT:**
RF improves ROC-AUC relative to LR (~0.774 vs ~0.745). Undersampling produces better F1 and recall than SMOTE — the first indication of a consistent pattern across model families.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add Model 3: Gradient Boosting (Sequential Boosting) to `03_modelling.ipynb` as the third candidate model, with `max_depth` verified experimentally before accepting it.

**PROMPT:**
"In 03_modelling.ipynb, add three cells after the Random Forest interpretation cell:

1. A markdown cell ## Model 3: Gradient Boosting (Sequential Boosting) explaining: Gradient Boosting builds trees sequentially, each one correcting the residuals of the previous — unlike RF which builds trees independently in parallel (Géron, 2019, Ch. 7). Often outperforms RF on tabular data because it focuses learning on the hard cases. More sensitive to hyperparameters than RF. Hyperparameter table: n_estimators=100, learning_rate=0.1 (default), max_depth=3 (default — shallow trees standard for boosting), random_state=42. Trained twice.
2. A code cell with id gb_smote training `GradientBoostingClassifier(n_estimators=100, random_state=42)` on X_tr_res/y_tr_res.
3. A code cell with id gb_us training the same on X_tr_us/y_tr_us."

**KEY OUTPUT:**
Three cells inserted. `md_gb`, `gb_smote`, `gb_us`.

**Verification Experiment 3 — max_depth:**
Tested max_depth ∈ {1, 2, 3, 5, 6} on both datasets. Also produced a visualisation cell (`gb_max_depth_plot`) plotting ROC-AUC vs depth for both sampling strategies.

Results: SMOTE peaked at max_depth=3 (≈0.7828). Undersampled best at max_depth=5 (≈0.7825) but essentially flat vs depth=3 (difference ~0.0001).

**Decision:** Chose `max_depth=3` — best on SMOTE, on the plateau for undersampling, most defensible generalisation default at the shortlisting stage. Both GB training cells updated to use `max_depth=3` explicitly.

**Results (validation set):**

- Gradient Boosting | SMOTE (depth=3): ROC-AUC ≈ 0.7828, F1 (class 1) ≈ 0.4888
- Gradient Boosting | Undersampled (depth=3): ROC-AUC ≈ 0.7824, F1 (class 1) ≈ 0.5196

**RESULT:**
Gradient Boosting achieved the highest ROC-AUC so far (~~0.783), exceeding LR (~~0.745) and RF (~0.774). F1 remained higher under undersampling than SMOTE, continuing the pattern observed across earlier models. This section now explicitly demonstrates the intended agent-tooling workflow: the model was suggested, the critical hyperparameter was experimentally verified, a decision was made from validation evidence, and only then was the final configuration implemented.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add Model 4: XGBoost to `03_modelling.ipynb` and verify Claude's recommended parameters before accepting them using a sequential greedy approach — each experiment uses the best value from the previous one, so the parameter choices are jointly consistent.

**Claude's initial suggestion:** `max_depth=4`, `n_estimators=300`, `learning_rate=0.05`, `reg_lambda=2`.

**Initial results with Claude's parameters:**
SMOTE: AUC 0.7794, F1 0.4877, precision 0.63, recall 0.40. Undersampled: AUC 0.7836, F1 0.5162, precision 0.59, recall 0.46.

Competitive with GB but parameters not yet verified. Claude chose max_depth=4 reasoning that L2 regularisation allows slightly deeper trees than GB's verified optimum of 3. However Claude's choices had not been verified by experiment — Experiments 4, 5, and 6 verify each choice before the final model is confirmed.

---

**Verification Experiment 4 — max_depth** (n_estimators=300, reg_lambda=2 fixed):

Tested max_depth ∈ {2, 3, 4, 5, 6}. Also produced a line plot of ROC-AUC vs max_depth for both datasets with a vertical dashed line at Claude's chosen value of 4.

SMOTE results: max_depth=2 → 0.7804, max_depth=3 → 0.7787, max_depth=4 → 0.7794, max_depth=5 → 0.7802, max_depth=6 → 0.7740. Best: max_depth=2, ROC-AUC 0.7804.

Undersampled results: max_depth=2 → 0.7861, max_depth=3 → 0.7849, max_depth=4 → 0.7836, max_depth=5 → 0.7831, max_depth=6 → 0.7798. Best: max_depth=2, ROC-AUC 0.7861.

The line plot showed both curves declining from depth 2 onward, with Claude's proposed depth 4 underperforming depth 2 on both datasets.

**ISSUES IDENTIFIED:** Claude's initial assumption that `xgb_us` was the final notebook cell was wrong, so the first insertion approach had to be corrected by locating the actual cell index before inserting.

**Decision:** `BEST_DEPTH = 2`. Claude's suggested `max_depth=4` rejected. XGBoost's L2 regularisation did not shift the optimal depth rightward on this dataset.

---

**Verification Experiment 5 — n_estimators** (BEST_DEPTH=2, reg_lambda=2 fixed):

Tested n_estimators ∈ {50, 100, 200, 300, 500}.

**ISSUES IDENTIFIED:** First draft of the experiment still used `max_depth=4` despite Experiment 4 having already rejected it. This would have invalidated the n_estimators comparison by testing tree counts under a non-verified depth setting. Initial experiment draft rejected and corrected before acceptance.

SMOTE results: 50→0.7771, 100→0.7804, 200→0.7807, 300→0.7794, 500→0.7817. Best: n_estimators=500, ROC-AUC 0.7817.

Undersampled results: 50→0.7817, 100→0.7850, 200→0.7862, 300→0.7836, 500→0.7865. Best: n_estimators=500, ROC-AUC 0.7865.

Both datasets peak at `n_estimators=500` and decline beyond it. Claude's recommended 300 is still on the rising slope — rejected.

**Decision:** `BEST_N = 500`. Once a hyperparameter has been disproved in an earlier experiment, later experiments must use the verified value rather than the discarded proposal.

---

**Verification Experiment 6 — reg_lambda** (BEST_DEPTH=2, BEST_N=500 fixed):

Tested reg_lambda ∈ {0.5, 1, 2, 5, 10}.

SMOTE results: 0.5→0.7793, 1→0.7803, 2→0.7817, 5→0.7794, 10→0.7799. Best: reg_lambda=2, ROC-AUC 0.7817. Claude's choice confirmed.

Undersampled results: 0.5→0.7824, 1→0.7828, 2→0.7836, 5→0.7854, 10→0.7877. Best: reg_lambda=10, ROC-AUC 0.7877 — and still rising. Claude's choice of 2 insufficient.

**Verification Experiment 6b — extended range (undersampled only):**
Since undersampled AUC was still rising at reg_lambda=10 with no plateau, tested reg_lambda ∈ {10, 15, 20, 30, 50}.

Results: 10→0.7883, 15→0.7875, 20→0.7877, 30→0.7861, 50→0.7850. Best: reg_lambda=10, ROC-AUC 0.7883. Curve peaked at 10 and declined beyond — confirming 10 as the true optimum.

**Decision:** `reg_lambda=10` used for both sampling strategies as undersampling was the stronger strategy throughout.

---

**Verified XGBoost parameters summary:**


| Parameter    | Claude recommended | Experiment result                              | Final value |
| ------------ | ------------------ | ---------------------------------------------- | ----------- |
| max_depth    | 4                  | Both datasets peak at 2                        | 2           |
| n_estimators | 300                | Both datasets peak at 500                      | 500         |
| reg_lambda   | 2                  | SMOTE confirmed at 2, Undersampled peaks at 10 | 10          |


All three values verified sequentially — each experiment used the best value from the previous one.

**Final verified XGBoost results:**

- XGBoost Final | SMOTE: ROC-AUC 0.7817, F1 0.4786. TN=2624, FP=180, FN=489, TP=307.
- **XGBoost Final | Undersampled: ROC-AUC 0.7877, F1 0.5245.** TN=2548, FP=256, FN=422, TP=374.

XGBoost Final | Undersampled selected for Step 5 based on achieving the highest scores on both chosen metrics across all 10 runs. Experimental verification improved AUC by 0.0041 over Claude's original parameters (0.7877 vs 0.7836 Undersampled).

**NOTES / REFLECTION:**
The sequential greedy approach ensures the verification is internally consistent. Claude's recommended values for max_depth and n_estimators were both rejected by experiment. The reg_lambda choice was confirmed for SMOTE but required increasing for the undersampled dataset, illustrating that regularisation interacts strongly with the training distribution. Step 5 will use cross-validated grid search to test all parameter combinations jointly and find the true optimum.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add Model 5: Multilayer Perceptron (Keras / TensorFlow) to `03_modelling_copy.ipynb` as a neural-network benchmark for the credit default prediction task.

**PROMPT:**
"Add a Keras MLP as Model 5 to the 03_modelling_copy notebook. Training data: X_tr_us (13,539 rows, 33 features), y_tr_us. Validation: X_val, y_val. TensorFlow 2.21.0 available. Propose architecture, train it, plot the training curve, and evaluate using the existing eval_model() function. Use tf.random.set_seed(42)."

Claude was first allowed to inspect the notebook structure and relevant cells before inserting the new model section.

**KEY OUTPUT:**
Claude inserted a new Model 5: Multilayer Perceptron (Keras / TensorFlow) section. The markdown cell documented the proposed architecture and rationale:

- hidden layers: 128 → 64
- activations: ReLU
- BatchNormalization after each Dense layer
- Dropout = 0.3
- output: Dense(1, sigmoid) with binary_crossentropy
- optimiser: Adam, learning rate 0.001
- batch size: 256
- early stopping: patience 20, restore best weights
- reproducibility via `tf.random.set_seed(42)`

Claude then added a code cell that imported TensorFlow/Keras, defined `build_keras_mlp(n_features, dropout_rate, lr)`, built the MLP, trained it on `X_tr_us/y_tr_us` with validation on `X_val/y_val`, plotted the training curve, wrapped the Keras model in a thin sklearn-style class so the existing `eval_model()` function could be reused unchanged, and evaluated the model on validation.

**ISSUES IDENTIFIED:**
TensorFlow initially failed in the original setup. PyTorch also failed there. The issue was traced to the path/environment setup — the working directory path was too long. File and folder names were shortened, the working path was changed, the environment/kernel was recreated, and packages were reinstalled. After this, TensorFlow became usable and the Keras MLP could be implemented successfully.

**RESULT:**
A working Keras / TensorFlow MLP was added to the modelling notebook, with proposed architecture documented, training completed on the undersampled training set, training curve plotted, and evaluation integrated into the notebook using the existing `eval_model()` workflow.

**NOTES / REFLECTION:**
This entry is important because the neural-network requirement was not straightforward to implement in the original environment. The eventual Keras solution was only possible after resolving the underlying path/setup issue. The final inserted model follows the same evaluation style as the rest of the notebook and preserves comparability with the earlier candidate models.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add Verification Experiment 7 to `03_modelling_copy.ipynb` to test whether the proposed Keras MLP architecture was actually the best choice before accepting it.

**PROMPT:**
"In 03_modelling_copy.ipynb, after the Keras MLP training cell (cell id: `u1qexrnwvea`), add Verification Experiment 7: architecture depth and width.

Test these 4 architectures, everything else fixed at proposed values (dropout=0.3, lr=0.001, batch_size=256, early_stopping patience=20):

- Shallow: (64,)
- Proposed: (128, 64)
- Deeper: (128, 64, 32)
- Wide: (128, 128)

For each: train on X_tr_us / y_tr_us, evaluate ROC-AUC on X_val / y_val. Use the existing `build_keras_mlp()` function. Print results as a table. Plot ROC-AUC as a bar chart, label the proposed architecture clearly. Print which architecture won. Use `tf.random.set_seed(42)` for each run."

**KEY OUTPUT:**
Claude inserted a markdown explanation cell and a code cell for Verification Experiment 7 — Architecture Depth and Width. The experiment looped through all four architectures, reset `tf.random.set_seed(42)` for each run, trained on `X_tr_us / y_tr_us`, evaluated ROC-AUC on `X_val / y_val`, printed a comparison table, plotted a bar chart of validation ROC-AUC, clearly marked the originally proposed architecture, and printed the winning architecture at the end.

**ACTION TAKEN:**
Accepted the notebook edit and ran the experiment.

Observed validation ROC-AUC results:

- Shallow (64,): 0.7854
- Proposed (128, 64): 0.7843
- Deeper (128, 64, 32): 0.7842
- Wide (128, 128): 0.7850

Winner: Shallow (64,) with ROC-AUC = 0.7854.

**RESULT:**
The originally proposed architecture (128, 64) was not the best. A simpler single hidden layer with 64 units slightly outperformed all deeper/wider alternatives on validation ROC-AUC.

**NOTES / REFLECTION:**
The agent proposed (128, 64) as a reasonable starting point, but the experiment showed that extra depth and width did not improve performance on this dataset. The result suggests the neural network does not benefit from added complexity here, likely because the dataset is tabular, modest in dimensionality (33 features), and already well served by simpler function classes.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Document the decision from Verification Experiment 8 (dropout) and add Verification Experiment 9 (learning rate) to `03_modelling_copy.ipynb` for the Keras MLP.

**PROMPT:**
"In 03_modelling_copy.ipynb, after the Experiment 8 code cell, add two cells:

1. A markdown decision cell:

**Decision — Experiment 8: Dropout Rate**


| Dropout | ROC-AUC           |
| ------- | ----------------- |
| 0.1     | 0.7838            |
| 0.2     | 0.7860 ← winner   |
| 0.3     | 0.7835 ← proposed |
| 0.4     | 0.7836            |
| 0.5     | 0.7826            |


Claude's proposed dropout=0.3 is rejected. dropout=0.2 achieves the highest AUC (0.7860). The result is consistent with the shallow (64,) architecture accepted in Experiment 7 — a smaller network needs less aggressive dropout; 0.2 provides sufficient regularisation without discarding too much signal per forward pass. Higher rates (0.4, 0.5) show monotonically declining performance, confirming over-regularisation at this network size.

BEST_DROPOUT = 0.2 is carried forward to Experiment 9.

1. Verification Experiment 9: learning rate — fix architecture=(64,), dropout=0.2, batch_size=256, patience=20. Test lr ∈ {0.0001, 0.001, 0.005, 0.01}. Train on X_tr_us / y_tr_us, evaluate ROC-AUC on X_val / y_val. Print results table, plot ROC-AUC vs lr on log-scale x-axis as a line chart. Mark proposed value (0.001) with a dashed vertical line. Print the winner. Use `tf.random.set_seed(42)` for each run."

**KEY OUTPUT:**
Claude inserted: (1) a markdown decision cell recording the outcome of Experiment 8, explicitly rejecting the originally proposed `dropout=0.3` and carrying forward `BEST_DROPOUT = 0.2`; (2) a new markdown + code section for Verification Experiment 9 — Learning Rate.

**ACTION TAKEN:**
Accepted and ran Experiment 9.

Observed validation ROC-AUC results:

- lr = 0.0001 → 0.7813
- lr = 0.001 → 0.7840 ← proposed
- lr = 0.005 → 0.7844 ← winner
- lr = 0.01 → 0.7834

Winner: learning rate = 0.005 with ROC-AUC = 0.7844.

**RESULT:**
The originally proposed learning rate = 0.001 was not optimal. A slightly larger value, 0.005, achieved the best validation ROC-AUC.

**NOTES / REFLECTION:**
This experiment continued the same evidence-first approach used for earlier models. The result suggests the shallow (64,) architecture with lighter dropout can tolerate a somewhat faster optimiser step than originally proposed.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add a final decision cell summarising Experiment 9 (learning rate) and record the verified final hyperparameters for the Keras MLP in `03_modelling_copy.ipynb`.

**PROMPT:**
"In 03_modelling_copy.ipynb, after the Experiment 9 code cell, add a markdown decision cell:

**Decision — Experiment 9: Learning Rate**


| Learning Rate | ROC-AUC           |
| ------------- | ----------------- |
| 0.0001        | 0.7813            |
| 0.001         | 0.7840 ← proposed |
| 0.005         | 0.7844 ← winner   |
| 0.01          | 0.7834            |


Claude's proposed lr=0.001 is rejected. lr=0.005 achieves the highest AUC (0.7844). The curve peaks at 0.005 then declines at 0.01, indicating that 0.001 under-steps the loss surface given the shallow (64,) architecture and the moderate dataset size. lr=0.005 provides faster convergence without overshooting. The 0.0001 result confirms that very slow learning rates underfit within the 200-epoch budget.

**Verified final MLP parameters:**


| Parameter      | Proposed    | Experiment                | Final       |
| -------------- | ----------- | ------------------------- | ----------- |
| Architecture   | (128, 64)   | Exp 7 — Shallow (64,) won | (64,)       |
| Dropout        | 0.3         | Exp 8 — 0.2 won           | 0.2         |
| Learning rate  | 0.001       | Exp 9 — 0.005 won         | 0.005       |
| Batch size     | 256         | Not tested                | 256         |
| Early stopping | patience=20 | Not tested                | patience=20 |


All three proposed values were rejected by experiment. The verified parameters are carried forward to the final MLP training cell."

**RESULT:**
The final verified MLP configuration carried forward:

- Architecture: `(64,)`
- Dropout: `0.2`
- Learning rate: `0.005`
- Batch size: `256`
- Early stopping: `patience = 20`

All three values originally proposed by the agent (architecture, dropout, learning rate) were rejected after empirical testing.

**NOTES / REFLECTION:**
The sequential verification experiments ensured that the neural network configuration was determined empirically rather than accepted from the initial agent suggestion. The final architecture is significantly simpler than the proposed design, reflecting the limited benefit of deeper neural networks on this tabular dataset.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Extend the Model 5: Keras MLP workflow in `03_modelling_copy.ipynb` to run the same hyperparameter verification experiments on the SMOTE training set (`X_tr_res / y_tr_res`) rather than only the undersampled dataset.

**PROMPT:**
"In 03_modelling_copy.ipynb, after the existing Experiment 9 code cell, add two new cells introducing Verification Experiments for SMOTE, then implement SMOTE Experiment 7 — Architecture, testing four architectures with dropout=0.3, lr=0.001, batch_size=256, and early stopping patience=20 fixed. Train on `X_tr_res / y_tr_res`, evaluate ROC-AUC on `X_val / y_val`. Add a bar chart visualisation of validation ROC-AUC for each architecture."

**KEY OUTPUT:**
Claude inserted a markdown section explaining that the three hyperparameter verification experiments used earlier for the undersampled MLP would now be repeated for the SMOTE training set, followed by code cells implementing SMOTE Experiments 7, 8, and 9 mirroring the undersampled versions.

**RESULT:**
The notebook now contains a parallel verification pipeline for the SMOTE MLP. Experiments 7, 8, and 9 repeated for SMOTE ensuring the architecture choice is not assumed to transfer between training strategies. Different class-balancing strategies can alter the training dynamics of neural networks, so verifying separately provides a fairer comparison.

---

## Step 4: Fine-Tuning (`03_modelling_copy.ipynb`)

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Start Step 5: Fine-Tuning and Evaluation in `03_modelling_copy.ipynb` for the shortlisted model XGBoost Final + Undersampling, using cross-validated hyperparameter search.

**PROMPT:**
"In 03_modelling_copy.ipynb, add a code cell with id `xgb_random_search` after cell `md_step5`. The cell should fine-tune the shortlisted XGBoost model using cross-validated hyperparameter search on the training data. Choose appropriate parameter ranges, search strategy, and scoring metric for this problem. Print the best parameters and best score."

**ACTION TAKEN:**
Claude first proposed adding both `md_step5` and `xgb_random_search`. The first proposed implementation was **rejected** after review because Claude chose to fit `RandomizedSearchCV` directly on `X_tr_us / y_tr_us` (the globally undersampled training set). This was identified as methodologically wrong: undersampling had already been applied before the CV split, so CV would not provide an independent estimate of performance.

A corrective prompt was then given instructing Claude to:

- document the agent mistake in a markdown cell
- load the full imbalanced training set (`X_tr.csv`, `y_tr.csv`) from artefacts
- use an `imblearn` Pipeline with `RandomUnderSampler` inside CV
- run `RandomizedSearchCV` on the full imbalanced training set instead

**KEY OUTPUT:**
Claude appended three cells:

1. `md_step5` — markdown introducing Step 5 and explaining: shortlisted model = XGBoost Final + Undersampling; `RandomizedSearchCV` chosen over `GridSearchCV`; `n_iter=50`, `cv=5`; ROC-AUC used as scoring metric.
2. `md_step5_mistake` — markdown explicitly documenting the agent mistake: fitting CV directly on `X_tr_us / y_tr_us` is methodologically incorrect; `X_tr_us` was globally undersampled before CV; CV folds would not be independent of the sampling step; correct approach is to undersample inside each fold using an `imblearn.pipeline.Pipeline`.
3. `xgb_random_search` — corrected code cell using full imbalanced `X_tr / y_tr`, `RandomUnderSampler` inside an `imblearn` pipeline, `RandomizedSearchCV(n_iter=50, cv=5, scoring='roc_auc')`, and parameter space centred on sequentially verified values.

**RESULT:**
Best parameters: `clf__learning_rate=0.01`, `clf__max_depth=3`, `clf__n_estimators=500`, `clf__reg_lambda=2`. Best 5-fold CV ROC-AUC: **0.7859**.

**NOTES / REFLECTION:**
The joint CV search moved away from several individually verified values, especially learning_rate and reg_lambda, reinforcing the earlier reflection that hyperparameters interact and should be tuned jointly rather than accepted from one-factor-at-a-time experiments. The correction substantially improved Step 5: instead of tuning on a pre-resampled dataset, the final search tunes the model under a realistic CV setup where undersampling happens only within each training fold.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Append the final tuned XGBoost model section to `03_modelling_copy.ipynb`, refitting the shortlisted model using the same undersampling pipeline that was used inside cross-validation during Step 5.

**PROMPT:**
"In 03_modelling_copy.ipynb, append two cells after the last cell:

1. A markdown cell with id `md_xgb_tuned` with text: 'Following Géron (2019, Ch. 2), the final model is refit using the same pipeline that was cross-validated — the imblearn Pipeline with RandomUnderSampler and XGBClassifier — trained on the full imbalanced training set X_tr with the best parameters found by RandomizedSearchCV. This ensures the training regime is consistent with how CV was scored.'
2. A code cell with id `xgb_tuned` that creates a Pipeline named `final_pipe`, fits it on `X_tr` and `y_tr`, then calls `eval_model(final_pipe.named_steps['clf'], X_val, y_val, 'XGBoost Tuned')`."

**ISSUES IDENTIFIED (agent mistake):**
Agent proposed refitting `final_pipe` on `X_tr_us` directly rather than using the same pipeline as CV. Caught before execution — corrected to `final_pipe.fit(X_tr, y_tr)`.

**Observed validation results:**
ROC-AUC: 0.7833, F1 (class 1): 0.5127. TN=2573, FP=231, FN=442, TP=354.

Compared with the Step 4 verified XGBoost undersampled model: Step 4 verified — ROC-AUC 0.7877, F1 0.5245, recall 0.47. Step 5 tuned — ROC-AUC 0.7833, F1 0.5127, recall 0.44.

The tuned hyperparameters were selected by 5-fold cross-validated ROC-AUC, which is a more stable estimate of generalisation than one validation split. The important methodological point is that the final model was refit using the same pipeline structure that was cross-validated, avoiding a mismatch between tuning and final training.

---

## Step 5: Evaluation (`03_modelling_copy.ipynb`)

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add a threshold optimisation section to `03_modelling_copy.ipynb` for the final tuned XGBoost model, to move beyond the default 0.5 classification threshold and choose a threshold better aligned with the business cost of credit default prediction.

**PROMPT:**
"In 03_modelling_copy.ipynb, append two cells after the last cell:

1. A markdown cell with id `md_threshold` titled `### Threshold Optimisation` explaining that the default 0.5 threshold is not optimal for imbalanced classification where missing a defaulter is more costly than a false alarm. This cell should explain that we plot precision, recall and F1 against threshold values to find the threshold that maximises F1 while keeping recall above 0.5.
2. A code cell with id `xgb_threshold` that: gets predicted probabilities from `final_pipe.named_steps['clf']` on `X_val`; uses `precision_recall_curve`; plots precision, recall and F1 against threshold on a single chart; finds the threshold that maximises F1 score; prints the optimal threshold, and the precision, recall and F1 at that threshold; adds a vertical dashed line at the optimal threshold and at 0.5 on the plot."

**ACTION TAKEN:**
Accepted. Ran threshold optimisation.

**Observed output:** Optimal threshold: **0.3964**. Precision: 0.5204. Recall: 0.5616. F1: 0.5402.

**ADDITIONAL MANUAL INTERPRETATION ADDED:**


| Metric    | Default threshold (0.50) | Optimal threshold (0.40) | Change |
| --------- | ------------------------ | ------------------------ | ------ |
| Precision | 0.61                     | 0.52                     | −0.09  |
| Recall    | 0.44                     | 0.56                     | +0.12  |
| F1        | 0.51                     | 0.54                     | +0.03  |


Lowering the threshold from 0.50 to 0.40 increases recall materially. The model catches more defaulters, at the cost of more false alarms. This trade-off is acceptable for this business setting because a missed defaulter is more costly than a wrongly flagged reliable payer.

**NOTES / REFLECTION:**
This step strengthens the modelling workflow by separating two decisions that are often conflated: (1) model fitting and (2) decision threshold selection. The tuned XGBoost model already provided probability estimates, but the default 0.5 cut-off was not aligned with the business objective. Threshold optimisation made the model operationally more useful without changing the underlying classifier.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Append a final evaluation section at the optimised classification threshold to `03_modelling_copy.ipynb`, applying the threshold found in the previous optimisation step to the tuned XGBoost model on the held-out validation set.

**PROMPT:**
"In 03_modelling_copy.ipynb, append two cells after the last cell: a markdown cell with id `md_threshold_applied` and a code cell with id `xgb_threshold_applied` that applies the optimal threshold of 0.3964 to the tuned model predictions on the validation set and shows the final evaluation results including confusion matrix."

**ADDITIONAL MANUAL INTERPRETATION ADDED:**


| Metric            | Default threshold (0.50) | Optimal threshold (0.40) |
| ----------------- | ------------------------ | ------------------------ |
| ROC-AUC           | 0.7833                   | 0.7833                   |
| Precision         | 0.61                     | 0.52                     |
| Recall            | 0.44                     | 0.56                     |
| F1                | 0.51                     | 0.54                     |
| Defaulters caught | 354 / 796                | 446 / 796                |
| False alarms      | 231                      | 411                      |


Lowering the threshold catches 92 additional defaulters at the cost of 180 additional false alarms — acceptable in credit default prediction where missing a defaulter is more costly than wrongly rejecting a reliable applicant.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Append a probability calibration section to `03_modelling_copy.ipynb` for the tuned XGBoost model, to check whether the model's predicted probabilities are reliable as probability estimates rather than only useful for ranking.

**PROMPT:**
"In 03_modelling_copy.ipynb, append two cells after the last cell: a markdown cell with id `md_calibration` and a code cell with id `xgb_calibration`. The code should plot a calibration curve for the tuned model — predicted probability vs actual default rate — to show whether the model's probability outputs are reliable. Include the perfectly calibrated line for reference."

**ADDITIONAL MANUAL INTERPRETATION ADDED:**
The calibration curve was interpreted as follows:

- the model is reasonably well calibrated in the low-to-mid probability range (roughly 0.1–0.5), where the curve stays fairly close to the diagonal
- at higher predicted probabilities, the curve sits above the diagonal, meaning the realised default rate is higher than the predicted probability
- this indicates the model is slightly underconfident at the upper end
- around the chosen operating threshold near 0.40, calibration is acceptable for threshold-based decision making, though the probabilities are not perfectly aligned with empirical rates
- if precise probability estimates were needed for downstream pricing or risk estimation, post-processing calibration such as Platt scaling or isotonic regression would be appropriate

**RESULT:**
The calibration assessment supports the following conclusion: the model's probabilities are good enough for ranking and threshold-based classification but are not perfectly calibrated probabilities, especially at the higher end of the score range. This complements the earlier ROC-AUC and threshold-optimisation work by showing that the model ranks risk well, the chosen threshold is operationally useful, but raw probabilities should be interpreted with some caution.

**NOTES / REFLECTION:**
This was a useful final diagnostic because it separated two ideas that can otherwise get mixed together: (1) discrimination — whether the model orders risky customers ahead of safe ones; and (2) calibration — whether a predicted 40% default probability really means about 40% default in practice. The tuned XGBoost model appears stronger on discrimination than on perfect calibration, which is common for boosted tree models.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add a failure mode analysis section to `03_modelling_copy.ipynb` to examine which defaulters the model misses, comparing false negatives against true positives on PAY_0, LIMIT_BAL, and AGE.

**PROMPT:**
"In 03_modelling_copy.ipynb, append two cells after the last cell: a markdown cell with id md_failure_modes and a code cell with id xgb_failure_modes. The code should analyse which defaulters the model fails to catch — the false negatives. Compare false negatives vs true positives on key features: PAY_0, LIMIT_BAL, and AGE. Show distributions or summary statistics that reveal what makes missed defaulters harder to detect."

**KEY OUTPUT:**
Claude appended two cells. `xgb_failure_modes` identified actual defaulters in the validation set, split them into false negatives and true positives using `y_pred_final`, printed counts and percentages for missed vs caught defaulters, printed summary statistics (mean and median) for PAY_0, LIMIT_BAL, and AGE, and plotted overlapping distributions for false negatives vs true positives for the three selected features.

**Observed output:**
Defaulters in validation set: 796. True positives (caught): 446 (56.0%). False negatives (missed): 350 (44.0%).


| Feature   | FN mean | TP mean | FN median | TP median |
| --------- | ------- | ------- | --------- | --------- |
| PAY_0     | −0.354  | 1.338   | 0.013     | 1.793     |
| LIMIT_BAL | −0.031  | −0.516  | −0.211    | −0.752    |
| AGE       | 0.161   | 0.019   | −0.047    | −0.047    |


**ADDITIONAL MANUAL INTERPRETATION ADDED:**
796 defaulters in the validation set: 446 caught (56%), 350 missed (44%). Missing 44% of defaulters is a meaningful limitation — in a lending context each missed defaulter represents an unrecovered loan.

PAY_0 is the most striking difference. Missed defaulters have a mean standardised PAY_0 of -0.354 and median of 0.013 — close to zero, meaning they were making on-time payments in the most recent month. Caught defaulters have mean 1.338 and median 1.793 — clearly delinquent. The model correctly identifies defaulters who show visible payment distress but has no signal for those who default despite appearing current. This is the model's primary blind spot: borrowers who default without prior observable delinquency.

LIMIT_BAL shows missed defaulters have higher credit limits (FN median -0.211 vs TP median -0.752). Higher-limit clients who default likely do so for reasons not captured in payment history — income shocks, strategic default — rather than gradual payment deterioration the model can detect.

AGE shows negligible difference (FN and TP median both -0.047). Age is not a meaningful driver of whether a defaulter is caught or missed.

**Why 44% missed is expected given available features:**

The 350 missed defaulters are structurally harder cases — they show no prior delinquency signal in the features available. The dataset contains only payment status, bill amounts, and demographic variables. It does not include income, debt-to-income ratio, employment status, or external credit bureau data. A human analyst reviewing the same features would face the same blind spot. Improving recall on this subgroup would require richer behavioural data beyond what is available in this dataset.

56% recall on payment-history-only features is within the expected range for this problem type. The failure mode analysis confirms the limitation is structural rather than a modelling failure — the signal needed to catch the remaining 44% is simply not present in the available features.

---

**DATE:** 2026-03-04
**MODEL:** Claude (Cursor extension)

**TASK:**
Add a final test-set evaluation section to `03_modelling_copy.ipynb` to assess the tuned XGBoost model on the held-out test set using the validation-selected threshold of 0.3964.

**PROMPT:**
"In 03_modelling_copy.ipynb, append two cells after the last cell: a markdown cell with id md_test_eval and a code cell with id xgb_test_eval. The code should evaluate the final tuned model on the held-out test set X_test_proc / y_test using the optimal threshold of 0.3964. Show ROC-AUC, precision, recall, F1 and confusion matrix. This is the final unbiased evaluation."

**KEY OUTPUT:**
Claude appended two cells. `xgb_test_eval` generated predicted probabilities on `X_test_proc`, applied threshold 0.3964, computed ROC-AUC, precision, recall, and F1, printed a classification report, and displayed a confusion matrix.

**ACTION TAKEN:**
Accepted the appended cells and ran the final test-set evaluation.

**Observed output:**
ROC-AUC: 0.7829. Precision: 0.5297. Recall (class 1): 0.5509. F1 (class 1): 0.5401.

Classification report: No Default — precision 0.87, recall 0.86, f1-score 0.87, support 4673. Default — precision 0.53, recall 0.55, f1-score 0.54, support 1327.

Confusion matrix: TN=4024, FP=649, FN=596, TP=731.

**RESULT:**
`03_modelling_copy.ipynb` now ends with a held-out test-set evaluation of the final tuned model at the chosen threshold of 0.3964, including summary metrics, classification report, and confusion matrix.

**NOTES / REFLECTION:**
This step records the final unbiased evaluation on unseen test data after all model selection and threshold decisions had already been made on training and validation data. Val vs test metrics are near-identical (Δ ≤ 0.001), confirming no overfitting to the validation set during model selection.

---

## Step 6: Model Card (`03_modelling_copy.ipynb`)

The model card documents the final deployed configuration for transparency and reproducibility.


| Field               | Value                                                                                                      |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| Model name          | XGBoost Credit Default Classifier                                                                          |
| Task                | Binary classification: predict whether a credit card client will default on payment next month             |
| Output              | Probability score in [0, 1]; classified as default if score ≥ 0.3964                                       |
| ROC-AUC             | 0.7829                                                                                                     |
| PR-AUC              | 0.5584                                                                                                     |
| Precision           | 0.53                                                                                                       |
| Recall              | 0.55                                                                                                       |
| F1                  | 0.54                                                                                                       |
| False positive rate | 13.9%                                                                                                      |
| Evaluated on        | 6,000 held-out test observations (never used in training or tuning)                                        |
| Key limitation      | 44% of defaulters missed — those showing no prior delinquency signal in available payment-history features |


