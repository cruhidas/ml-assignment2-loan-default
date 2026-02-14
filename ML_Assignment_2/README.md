# ML Assignment 2 - Loan Default Prediction (Lending Club)

## a) Problem statement
Build multiple classification models to predict whether a loan will default (Charged Off) or not (Fully Paid).
Also create a Streamlit app to demo model selection + metrics.

## b) Dataset description
Dataset: Lending Club loan data (2007-2014)
File used: `loan_data_2007_2014_New.csv`
Target:
- Fully Paid -> 0
- Charged Off -> 1
We removed other loan_status values like Current etc.

Min requirements (from assignment): >=12 features and >=500 rows (this dataset satisfies after cleaning).

## c) Models used + Metrics table
Models (6):
1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes (Gaussian)
5. Random Forest
6. XGBoost

Metrics calculated:
Accuracy, AUC, Precision, Recall, F1, MCC

After running `train_models.py`, the table is saved at:
`saved_models/metrics_table.csv`

(put the table here in final submission by copying the printed output or csv)

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
|Logistic Regression|0.9915254237288136|0.999838739017783|0.9976635514018691|0.9510022271714922|0.9737742303306728|0.9691190774636078|
|Random Forest|0.9922623434045689|0.9995113005599886|1.0|0.9532293986636972|0.976054732041049|0.9718398521109528|
|XGBoost|0.9963154016212233|0.99914354685664|1.0|0.977728285077951|0.9887387387387387|0.9866258552918403|
|Decision Tree|0.9882092851879145|0.981329124815017|0.9582417582417583|0.9710467706013363|0.9646017699115044|0.9575591197064669|
|Naive Bayes|0.9609432571849669|0.9649635933666671|0.9341772151898734|0.821826280623608|0.8744075829383886|0.8538459350842165|
|KNN|0.8802505526897568|0.7985958494963054|0.8563218390804598|0.33184855233853006|0.478330658105939|0.4866481488631681|


### Observations (add after you see results)
| ML Model | Observation about performance |
|---|---|
| Logistic Regression | Achieved very high Accuracy and AUC, indicating strong linear separability in the loan dataset. Performed surprisingly close to ensemble models after scaling. |
| Decision Tree | Performed well but slightly lower than ensemble models. May overfit individual splits and shows higher variance compared to Random Forest and XGBoost. |
| KNN | Lowest Recall and MCC among all models. Performance likely affected by high dimensionality and sensitivity to feature scaling. Not ideal for this dataset. |
| Naive Bayes | Moderate performance with good speed. Assumes feature independence, which may not fully hold in financial loan data. |
| Random Forest | Very strong performance with near-perfect AUC and high MCC. Handles nonlinear relationships and reduces overfitting compared to Decision Tree. |
| XGBoost | Achieved extremely high AUC and F1 score. Ensemble boosting approach provides robust and stable classification performance for loan default prediction. |


## How to run (BITS Lab)
1) Put `loan_data_2007_2014_New.csv` in the same folder as `train_models.py`
2) Install deps:
```
pip install -r requirements.txt
```
3) Train + save models:
```
python train_models.py
```
4) Run Streamlit:
```
streamlit run app.py
```

## Links for submission (fill before final PDF)
- GitHub Repo Link: <paste here>
- Streamlit App Link: <paste here>

Notes:
- Streamlit upload should be test data only (small) because free tier is limited.
- For app metrics, include a `target` column in the uploaded CSV.
