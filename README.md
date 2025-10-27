# 🛒 Blinkit Retention Project

**Short Abstract:**  
This project implements a **segmentation and churn prediction pipeline** for a Blinkit-like e-grocery dataset. It combines **Recency-Frequency-Monetary (R+FMD) segmentation** with **sentiment-aware churn prediction**, helping businesses identify customers at risk of leaving and understand behavioral patterns.

---

## ⚡ Quick Setup

```bash
git clone https://github.com/<your-username>/churnaware.git
cd churnaware
python -m venv venv
python -m pip install --upgrade pip
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac
python -m pip install -r requirements.txt
python -m streamlit run src/dashboard.py
