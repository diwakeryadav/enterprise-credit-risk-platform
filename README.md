# Enterprise Credit Risk Prediction Platform 

An enterprise-grade, end-to-end Machine Learning and MLOps system to predict loan default probability, structured to comply with Tier-1 banking regulations (such as **Federal Reserve SR 11-7** for Model Risk Management and **BCBS 239** for Risk Data Aggregation).

This repository is designed to simulate a real-world internal credit scoring system, utilizing robust data validation, feature engineering, explainable AI (SHAP), experiment tracking (MLflow), automated unit tests (Pytest), and FastAPI containerized serving (Docker).

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
* **Feature Drift Monitoring**: Implements Kolmogorov-Smirnov (KS) drift testing between reference training sets and baseline/serving data.

### 3. Explainable AI and Fair Lending (ECOA / Regulation B)
* **Adverse Action Reasons**: Generates local SHAP values alongside prediction probabilities, returning the top 3 specific reasons for loan denial to comply with the Equal Credit Opportunity Act (ECOA).

---

## 🛠️ Project Architecture

```
enterprise-credit-risk-platform/
├── configs/                  # Central YAML configurations and schemas
│   ├── config.yaml
│   └── schema.yaml
├── data/                     # Local data storage (raw and transformed splits)
│   ├── raw/
│   └── processed/
├── logs/                     # Rotated persistent log files
├── models/                   # Serialized ML model binaries
├── artifacts/                # Stage-to-stage pipeline outputs
├── reports/                  # Validation reports and drift stats
├── notebooks/                # Exploratory Data Analysis & experiments
├── tests/                    # Automation test suite (Pytest)
│   ├── test_api.py           # API integration tests
│   └── test_config.py        # Config schema structure checks
├── src/                      # Source code directory
│   ├── api/                  # FastAPI serving layer (Routes and dashboard UI)
│   │   ├── routes/           # Explainability and monitoring API endpoints
│   │   └── templates/        # Dashboard HTML/CSS templates
│   ├── components/           # Pipeline components (Ingestion, Validation, Transformation, etc.)
│   ├── config/               # Configuration loading & directory setups
│   ├── constants/            # Global constants and file paths
│   ├── entity/               # Typed frozen configuration and artifact dataclasses
│   ├── exception/            # Structured traceback custom exceptions
│   ├── explainability/       # Multi-agent LangGraph system for compliance explainability
│   ├── logger/               # Rotated file and stream logger utilities
│   ├── pipeline/             # Training and prediction pipelines
│   └── utils/                # Standard file and binary serialization utilities
├── requirements.txt          # Python pinned packages
├── test_exception.py         # Exception verification script
├── test_data_ingestion.py    # Temporary data ingestion driver
├── Dockerfile                # API container script
├── docker-compose.yml        # Orchestration script
└── README.md                 # Project documentation
```

---

## 🚀 Current Project Status & Completed Modules

### Phase 1: Foundation (Completed)
We have successfully established the production foundation for the project:
1. **Pinned Dependency Management** ([requirements.txt](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/requirements.txt)): Configured strict core libraries, serving frameworks, experiment trackers, and test suites.
2. **Centralized Configurations** ([configs/config.yaml](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/configs/config.yaml)): Hierarchical configs separating parameters across pipeline stages.
3. **Structured Rotated Logging** ([src/logger/](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/logger/)): Logs simultaneously to the terminal (for container runs) and persistent timestamped files under `/logs`.
4. **Detailed Traceback Exceptions** ([src/exception/](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/exception/)): Captures exact source code filenames, line numbers, and error details, preventing swallowed exceptions.
5. **Common File Utilities** ([src/utils/](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/utils/)): Implemented safe, robust binary serialization using `dill` (capable of pickling complex data transformers) along with JSON and YAML helper functions.
6. **Domain Entities** ([src/entity/](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/entity/)): Configured immutable (`frozen=True`) dataclasses for configurations and artifact delivery, ensuring strict input/output contracts between pipeline stages.
7. **Configuration Manager** ([src/config/](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/config/)): Centralized parser that loads configs and automatically bootstraps all directories at pipeline runtime.

### Phase 2: Core Components & Interactive API Serving (Completed)
We have designed and developed the core risk validation components, regulatory-compliant AI models, and real-time dashboard API:
1. **Data Validation Component** ([src/components/data_validation.py](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/components/data_validation.py)): Implements `DataDriftDetector` that utilizes **Evidently AI** to perform feature drift detection (e.g. using Kolmogorov-Smirnov test statistics) between reference datasets (training baseline) and target datasets. Generates audit-compliant JSON reports.
2. **Explainability Agent** ([src/explainability/agent.py](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/explainability/agent.py)): Built a **LangGraph-based multi-agent workflow** to generate compliant loan denial explanations from SHAP feature importances and applicant data.
   - **SHAP Analysis Node**: Extracts the top 3 features contributing to default risk.
   - **Business Explanation Node**: Drafts a business-friendly, plain English narrative using LLMs (`gpt-4o-mini`).
   - **Compliance Review Node**: Audits the draft against FCRA/ECOA rules. Loops to revise the narrative if compliance criteria are not met.
3. **FastAPI Serving Layer** ([src/api/](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/api/)):
   - `/health`: Health status endpoint.
   - `POST /api/v1/explainability/explain-denial`: Exposes the LangGraph explainability workflow to generate compliant denial explanations.
   - `POST /api/v1/monitoring/check-drift`: Triggers feature drift detection on batch inference data.
   - `POST /api/v1/monitoring/setup-demo`: Utility endpoint to automatically generate synthetic baseline and clean/drifted batch datasets.
4. **Interactive Dashboard** ([src/api/templates/index.html](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/src/api/templates/index.html)): A dark-themed, glassmorphic UI served at `/` to let users load preset loan profiles, evaluate applicant data against compliance agents, visualize agent state progression, and trigger data drift checks.
5. **Test Automation** ([tests/](file:///c:/Users/diwak/OneDrive/Desktop/personal/enterprise-credit-risk-platform/tests/)): Implemented comprehensive integration tests under `tests/test_api.py` and `tests/test_config.py` using `pytest`.

---

## ⚙️ Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/diwakeryadav/enterprise-credit-risk-platform.git
cd enterprise-credit-risk-platform
```

### 2. Create and Activate Virtual Environment
On Windows:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify the Environment Foundation
To test the custom exceptions, logging outputs, and configuration parsing, run:
```bash
# Verify custom exception formatting and logging
python test_exception.py

# Verify configuration parser parses yaml and boots folders
python -c "from src.config import ConfigurationManager; cm = ConfigurationManager(); print(cm.get_data_ingestion_config())"
```

### 5. Running the Application and UI Dashboard
You can run the FastAPI server locally:
```bash
python src/api/main.py
```
* **Dashboard Access**: Open `http://localhost:8000/` in your browser.
* **API Documentation (Swagger UI)**: Open `http://localhost:8000/docs`.

### 6. Running Automated Tests
To run unit and integration tests:
```bash
python -m pytest
```
