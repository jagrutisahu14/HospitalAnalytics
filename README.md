# Hospital Management Data Analytics & Predictive Dashboard

**Author:** Jagruti Sahu
**Program:** IBM SkillsBuild Data Analytics with AI Internship 2026 (BharatCares / AICTE)
**Built with:** Python and IBM Bob (AI development partner)

## 1. Project description
An end-to-end data analytics project on a hospital management database. It combines five tables into one analysis-ready dataset, calculates operational and financial KPIs, shows them in an interactive dashboard, and tests three machine-learning models and a revenue forecast.

## DataSet link - https://www.kaggle.com/datasets/kanakbaghel/hospital-management-dataset

**Business questions answered**
- How many appointments are completed, cancelled or no-shows?
- Which treatments, specializations and doctors bring the most revenue?
- How much billed money is collected, pending or failed?
- Can bill amount, appointment no-shows and payment status be predicted?
- What will billed revenue look like in the next 6 months?

## 2. Dataset
Five CSV files (patients, doctors, appointments, treatments, billing) in the standard Hospital Management schema.
**Dataset link:** <ADD YOUR DATASET LINK HERE>

| File | Key columns |
|---|---|
| patients.csv | patient_id, gender, date_of_birth, insurance_provider |
| doctors.csv | doctor_id, specialization, years_experience, hospital_branch |
| appointments.csv | appointment_id, patient_id, doctor_id, appointment_date, reason_for_visit, status |
| treatments.csv | treatment_id, appointment_id, treatment_type, cost |
| billing.csv | bill_id, patient_id, treatment_id, amount, payment_method, payment_status, bill_date |

## 3. Technologies used
Python 3, Pandas, NumPy, Matplotlib, Scikit-learn, Statsmodels, Plotly, Dash, Jupyter Notebook, IBM Bob.

## 4. Project files
| File | Purpose |
|---|---|
| `Jagruti_Sahu_HospitalAnalytics.ipynb` | Main code: load, clean, merge, KPIs, charts, 3 ML models, forecast |
| `dashboard.py` | Interactive Dash dashboard (frontend) with filters |
| `requirements.txt` | Python dependencies |
| `Jagruti_Sahu_ProjectReport.docx` | Full project report |

## 5. Setup and run
1. Install Python 3.10 or newer.
2. Put the 5 CSV files in the same folder as the code files.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook: `jupyter notebook Jagruti_Sahu_HospitalAnalytics.ipynb`, then **Run All**.
5. Run the dashboard:
   ```bash
   python dashboard.py
   ```
   Open http://127.0.0.1:8050 in a browser.

## 6. What the project does
1. **Data quality checks:** nulls, duplicate IDs and orphan foreign keys, followed by cleaning.
2. **Master table:** appointments joined with patients, doctors, treatments and billing.
3. **KPIs:** completion, cancellation and no-show rates, total billed, collected, pending and failed amounts, collection rate.
4. **Dashboard:** filters for date range, specialization, branch and payment status; KPI cards and 6 charts.
5. **Models:** bill amount regression, no-show classification, and payment status classification. Each is compared with a baseline and cross-validated.
6. **Forecast:** 6-month billed revenue forecast with Linear trend, Holt-Winters and Random Forest, backtested on the last 3 months against a mean baseline.

## 7. Key findings and limitations
- A large share of booked appointments end as cancellations or no-shows, and only part of billed revenue is collected. Reminders and payment follow-ups are the main recommendations.
- Model results are reported honestly with baselines. On small datasets (a few hundred rows), models can perform close to chance level, so results should be treated as a working prototype and not used for real decisions without more data and richer features (booking lead time, payment history, procedure codes).
- Forecasts use about one year of monthly data, so they are rough planning ranges only.
- Real patient data must be anonymised and handled under privacy regulations such as HIPAA.
