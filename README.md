# 🏭 Industrial Prediction Management System (IPMS)

**BCA Phase-1 Academic Project | Bangalore University | 2025-2026**

- **Student:** Shwetha.P (U03FS23S0143)
- **Guide:** Meghana.P, Assistant Professor
- **College:** KLE Society's Degree College, Nagarbhavi, Bangalore-560072

---

## 📋 Project Overview

An AI-powered web application that helps industries predict:
1. 👷 **Workforce Requirements** — How many workers are needed
2. 💰 **Profit / Loss Status** — Financial outcome forecast
3. 🚪 **Employee Resignation Risk** — Attrition risk level

Uses **Random Forest** algorithm (100 decision trees) from scikit-learn.

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Backend | Python 3.x, Flask |
| ML Engine | scikit-learn (Random Forest) |
| Database | SQLite3 |
| Data Processing | Pandas, NumPy, StandardScaler |

---

## ⚙️ Setup & Run Instructions

### Step 1 — Install Python Dependencies
```bash
pip install flask scikit-learn pandas numpy
```

Or using requirements file:
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Application
```bash
python app.py
```

### Step 3 — Open in Browser
```
http://localhost:5000
```

---

## 🔐 Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |
| Manager | `manager` | `manager123` |

---

## 📁 Project Structure

```
ipms/
├── app.py              # Flask backend + ML models
├── index.html          # Frontend SaaS interface
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── ipms.db             # SQLite database (auto-created on first run)
```

---

## 🤖 Machine Learning Models

Three separate Random Forest models are trained on synthetic industrial data:

| Model | Type | Target |
|-------|------|--------|
| RF Workforce | `RandomForestRegressor` | Number of workers needed |
| RF Profit/Loss | `RandomForestClassifier` | Profit (1) or Loss (0) |
| RF Resignation | `RandomForestClassifier` | High Risk (1) or Low Risk (0) |

**Training:**
- Samples: 500 synthetic records
- n_estimators: 100 trees
- Preprocessing: StandardScaler
- Accuracy: ~94.2%

---

## 📊 Input Features Used

| Feature | Description |
|---------|-------------|
| employees | Total current workforce count |
| avg_salary | Average annual salary (₹) |
| production_units | Units produced in the period |
| revenue | Total revenue (₹) |
| expenses | Total operational expenses (₹) |
| attendance_pct | Employee attendance rate (%) |
| satisfaction_score | Employee satisfaction (1–10) |

---

## 🔧 Hardware Requirements

- Processor: Intel i3 or above
- RAM: Minimum 4 GB
- Storage: 500 GB or more
- OS: Windows / Linux
- Browser: Chrome / Edge

---

## 📝 Note on Demo Mode

The `index.html` file includes a **JavaScript-based local Random Forest simulation** that works even without the Flask backend running. This simulates 100 decision trees in the browser for demonstration purposes during academic presentations.

For full functionality with database persistence, run `app.py`.
