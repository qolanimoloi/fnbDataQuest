# FNB DataQuest 2026 — Credit Intelligence Platform
### Moloi Qolani Truelove & Tshegofatso Tshepang Chikwane — Sol Plaatje University

---

## What this is

A full 6-page Streamlit application built for the FNB DataQuest 2026 competition.
It hosts the complete credit risk intelligence suite on top of the trained
WoE logistic regression model.

## Pages

| Page | What it does |
|------|-------------|
| 🏠 Home | Project overview, live model metrics, team details |
| 📊 Data Quality | Missing values, class imbalance, outlier detection, column summary |
| 🔬 EDA Explorer | Interactive univariate & bivariate explorer, WoE/IV rankings |
| 📈 Business Dashboard | Threshold slider, Rand-value impact, Precision/Recall business meaning |
| 🤖 Credit AI Chatbot | Claude-powered chatbot — ask anything about the model in plain English |
| 📄 Loan Decision Report | Score any applicant, generate approved/declined HTML report with repayment timeline |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Make sure `DataQuest26/DataQuest26/loan_book.csv` is accessible from the folder
where you run the app. The path is relative to your working directory.

```
your_project/
├── app.py
├── model_engine.py
├── report_generator.py
├── requirements.txt
└── DataQuest26/
    └── DataQuest26/
        └── loan_book.csv
```

### 3. Set your Claude API key (for the chatbot)
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at http://localhost:8501

---

## Notes

- The model trains automatically on first load and is cached — subsequent page
  switches are instant.
- The chatbot requires a valid `ANTHROPIC_API_KEY` environment variable.
  Without it, all other pages work fully.
- Decision reports download as standalone HTML files — no internet connection
  required to open them.
- The app uses only the training split for WoE fitting to prevent data leakage.

---

*FNB DataQuest 2026 — "From Roots to Rise"*
