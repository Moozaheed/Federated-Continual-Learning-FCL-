# Action Plan: Immediate Next Steps

## 1. Environment Setup (Today)
- [ ] Create a virtual environment: `python -m venv venv`
- [ ] Install dependencies: `pip install torch flwr pandas numpy matplotlib scikit-learn`
- [ ] Initialize a git repository.

## 2. Data Acquisition (Target: Tomorrow)
- [ ] Write `download_data.py`:
    - Download `processed.cleveland.data`
    - Download `processed.hungarian.data`
    - Download `processed.switzerland.data`
    - Download `processed.va.data` (Long Beach)
- [ ] Write `preprocess.py`:
    - Handle the '?' characters (missing values).
    - Normalize numerical features.
    - Save clean CSVs: `cleveland_clean.csv`, etc.

## 3. First Script
- [ ] Create a simple `eda.ipynb` or script to visualize the **Drift** (show how different the age/cholesterol distribution is between Cleveland and Switzerland).

## 4. Federated Skeleton
- [ ] Create `server.py` and `client.py` using the Flower quickstart template.
