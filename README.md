# Enterprise Credit Risk Prediction Platform 

An enterprise-grade, end-to-end Machine Learning and MLOps system to predict loan default probability, structured to comply with Tier-1 banking regulations (such as **Federal Reserve SR 11-7** for Model Risk Management and **BCBS 239** for Risk Data Aggregation).

This platform combines traditional machine learning with a **DeepSeek-R1 multi-agent compliance explainability system**, automated data drift detection (**Evidently AI**), and an interactive glassmorphic dashboard built with FastAPI.

---

## 🤖 Active AI Engine: DeepSeek-R1 (Local & Zero-Cost)

The compliance explainability layer in this project runs **DeepSeek-R1** (`deepseek-r1:1.5b` or `deepseek-r1:8b`) locally via **Ollama**. 

* **Zero API Costs**: Runs entirely on local hardware with zero subscription fees or OpenAI quota limits.
* **OpenAI API Compatibility**: Standardized endpoint integration via `http://localhost:11434/v1`.
* **Advanced Chain-of-Thought Reasoning**: DeepSeek-R1 excels at step-by-step compliance auditing, protected class checks, and regulatory narrative synthesis under ECOA/FCRA guidelines.

---

## 🏛️ Regulatory and Compliance Standards

In consumer credit risk modeling, models cannot be simple black boxes due to strict global regulatory guidelines. This platform incorporates compliance metrics into its core architecture:

### 1. Model Risk Management (SR 11-7 / OCC 2011-12)
* **Conceptual Soundness**: Features are constructed to reflect real financial risk dimensions (e.g., Debt-to-Income, Annuity-to-Income, and Payment-to-Income).
* **Comprehensive Performance Suite**: Rather than focusing solely on ROC-AUC, the evaluation pipeline computes:
  * **PR-AUC (Precision-Recall AUC)**: Evaluates predictions under heavy class imbalance (~8% default rate).
  * **Kolmogorov-Smirnov (KS) Statistic**: Measures the maximum separation between defaults and non-defaults.
  * **Financial Cost Matrix Optimization**: Thresholds are determined by balancing default loss (False Negatives) against customer opportunity loss (False Positives).

### 2. Risk Data Aggregation and Lineage (BCBS 239)
* **Strict Schema Conformity**: Raw data must pass structured schemas (valid columns, correct types, value ranges).
* **Feature Drift Monitoring**: Implements Kolmogorov-Smirnov (KS) drift testing using **Evidently AI** between reference training sets and baseline/serving data.

### 3. Explainable AI and Fair Lending (ECOA / Regulation B)
* **Adverse Action Reasons**: Generates local SHAP values alongside prediction probabilities, returning the top 3 specific reasons for loan denial to comply with the Equal Credit Opportunity Act (ECOA).
* **Multi-Agent Compliance Auditing**: Powered by a **LangGraph** multi-agent graph running **DeepSeek-R1** to generate and iteratively audit adverse action explanations against regulatory guidelines.

---

## 🛠️ Project Architecture

```
enterprise-credit-risk-platform/
├── configs/                  # Central YAML configurations and schemas
│   ├── config.yaml           # DeepSeek-R1 model params, drift thresholds & agent configs
│   └── schema.yaml           # Config structure validation schema
├── data/                     # Local data storage (raw and processed splits)
│   ├── raw/
│   └── processed/
├── logs/                     # Rotated persistent log files
├── models/                   # Serialized ML model binaries
├── artifacts/                # Stage-to-stage pipeline outputs
├── reports/                  # Validation reports and drift stats
├── scripts/                  # Utility and validation scripts
│   └── validate_yaml.py      # Automated YAML schema validator
├── tests/                    # Automation test suite (Pytest)
│   ├── test_api.py           # API integration tests
│   └── test_config.py        # Config schema structure checks
├── src/                      # Source code directory
│   ├── api/                  # FastAPI serving layer & UI dashboard
│   │   ├── routes/           # Explainability and monitoring API endpoints
│   │   └── templates/        # Glassmorphic UI Dashboard (index.html)
│   ├── components/           # Pipeline components (Ingestion, Validation, etc.)
│   ├── config/               # Configuration parser & directory setup
│   ├── constants/            # Global constants and file paths
│   ├── entity/               # Typed frozen configuration & artifact dataclasses
│   ├── exception/            # Custom traceback exception handling
│   ├── explainability/       # LangGraph multi-agent system (DeepSeek-R1 engine)
│   ├── logger/               # Rotated file & stream logging utilities
│   ├── pipeline/             # Training and prediction pipelines
│   └── utils/                # Serialization (dill/YAML/JSON) helpers
├── .env                      # Local environment variable configuration
├── requirements.txt          # Pinned dependency manifest
├── Dockerfile                # API container script
├── docker-compose.yml        # Multi-container orchestration
└── README.md                 # Project documentation
```

---

## 🚀 Key Features & Completed Modules

### Phase 1: Foundation & Infrastructure
1. **Pinned Dependency Management** ([requirements.txt](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/requirements.txt)): Core ML libraries, LangGraph, FastAPI, Evidently AI, and Pytest.
2. **Centralized Configuration System** ([configs/config.yaml](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/configs/config.yaml)): Central YAML settings configured for `deepseek-r1:1.5b`.
3. **Structured Rotated Logging** ([src/logger/](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/logger/)): Dual console and persistent log file output under `/logs`.
4. **Detailed Traceback Exceptions** ([src/exception/](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/exception/)): Captures exact Python filenames, line numbers, and trace details.
5. **Configuration Validator** ([scripts/validate_yaml.py](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/scripts/validate_yaml.py)): Automated YAML syntax and key completeness verification script.

### Phase 2: Core Risk Components & Interactive Dashboard
1. **Data Validation Component** ([src/components/data_validation.py](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/components/data_validation.py)): Feature drift detector powered by **Evidently AI** with JSON audit report outputs.
2. **DeepSeek-R1 Explainability Agent** ([src/explainability/agent.py](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/explainability/agent.py)): Multi-agent **LangGraph** workflow for adverse action notices:
   - **SHAP Analysis Node**: Ranks and extracts the top 3 default risk factors.
   - **Business Explanation Node**: Drafts plain-English denial narratives using **DeepSeek-R1**.
   - **Compliance Audit Node**: Audits text against FCRA/ECOA rules to guarantee protected class exclusion and objectivity.
3. **FastAPI Web Service** ([src/api/main.py](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/api/main.py)): REST endpoints exposing health status, drift detection, demo dataset generation, and denial explainability.
4. **Interactive Dashboard** ([src/api/templates/index.html](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/api/templates/index.html)): Glassmorphic dashboard to load loan profiles, evaluate applicant data against AI compliance agents, and run statistical data drift checks.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/diwakeryadav/enterprise-credit-risk-platform.git
cd enterprise-credit-risk-platform
```

### 2. Set Up Virtual Environment
On Windows:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull Local DeepSeek-R1 Model via Ollama
Ensure [Ollama](https://ollama.com/) is installed and running, then pull the DeepSeek-R1 model:
```bash
ollama pull deepseek-r1:1.5b
```

---

## 🚀 Running the Application

### 1. Validate Configurations
```bash
python scripts/validate_yaml.py
```

### 2. Run Automated Test Suite
```bash
python -m pytest
```

### 3. Launch FastAPI Application & Dashboard
Execute the application as a Python module from the project root:
```bash
python -m src.api.main
```

* **Interactive Dashboard**: Open `http://localhost:8001/` (or `http://localhost:8000/`) in your browser.
* **Swagger API Docs**: Open `http://localhost:8001/docs`.
