# Blinkit Retention Project

This repository implements the R+FMD segmentation and sentiment-aware churn prediction pipeline for Blinkit-like e-grocery dataset.

## Project structure
(see full structure in README)

## Quickstart
1. Put the dataset CSVs in the `data/` folder (names must match).
2. Create a Python environment and install requirements:

pip install -r requirements.txt

3. Run segmentation pipeline:


python src/train_segmentation.py

4. Run churn training:


python src/train_churn.py

5. Run the dashboard:


streamlit run src/dashboard.py


## Notes
- Default churn threshold = 90 days. Changeable in `train_churn.py` or `Mymodules/modeling.py`.
- Segmentation uses GaussianMixture (GMM). Adjust components via CLI or script.

# Clone repo
git clone https://github.com/PrasannaMishra001/ChurnAware.git
cd ChurnAware

# Create virtual environment
python -m venv venv
# Activate venv (Windows)
venv\Scripts\activate
# Activate venv (Mac/Linux)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run src/dashboard.py
