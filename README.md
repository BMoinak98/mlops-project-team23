# Telco Customer Churn MLOps Platform - Team 23

An end-to-end industrial MLOps platform built to orchestrate, train, track, deploy, and monitor PySpark Machine Learning models for predicting Telco Customer Churn.

---

## 👥 Team Members

* **Moinak Bandyopadhyay** (DA25M591)
* **Rahul Reddy** (DA25M609)

---

## 🏗 System Architecture & Workflow


```

┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Data Ingestion  │ ──> │ PySpark ML Pipeline │ ──> │ MLflow Tracking  │
│  (Parquet/S3)   │     │  (Preprocessing) │     │ (Params/Metrics) │
└─────────────────┘     └──────────────────┘     └──────────────────┘
│
┌─────────────────┐     ┌──────────────────┐              ▼
│  Prometheus &   │ <── │ FastAPI Serving  │ <── ┌──────────────────┐
│ Grafana Metrics │     │  (Inference Endpoint)  │ │ MLflow Model     │
└─────────────────┘     └──────────────────┘     │    Registry      │
└──────────────────┘

```

1. **Orchestration**: Apache Airflow triggers data preparation and distributed PySpark model training pipelines.
2. **Data & Training**: PySpark handles class rebalancing, indexing, encoding, and training (Logistic Regression & Random Forest).
3. **Experiment Tracking**: MLflow logs metrics, parameters, and model artifacts using proxied artifact storage (`--serve-artifacts`).
4. **Model Serving**: FastAPI loads trained PySpark pipeline models directly from MLflow for real-time REST API inference.
5. **Observability**: Prometheus captures latency and prediction metrics, visualized in Grafana dashboards.
6. **CI/CD Pipeline**: GitHub Actions automates container builds (GHCR), remote directory provisioning, and live service deployments.

---

## 📂 Project Structure

```directory
.
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD deployment pipeline (server setup, image builds, code sync)
├── config/
│   ├── config-docker.yml         # Application and ML pipeline configuration
│   └── prometheus.yml            # Prometheus metric scraper configuration
├── dags/                         # Airflow DAG workflow definitions
├── docker/
│   ├── Dockerfile.airflow        # Custom Airflow runtime with Java and PySpark support
│   └── Dockerfile.api            # FastAPI container runtime configuration
├── src/
│   ├── api/
│   │   └── main.py               # FastAPI inference application & PySpark pipeline loading
│   └── data_engineering/
│       ├── common_util.py        # Config parsing and helper utilities
│       ├── data_preprocess.py    # Feature engineering and ETL preprocessing
│       └── model_train.py       # Distributed PySpark ML training & MLflow logging
├── docker-compose.yml            # Infrastructure service orchestration
└── README.md

```

---

## 🌐 Service Port & Endpoint Matrix

| Service | Host Port | Target URL | Description |
| --- | --- | --- | --- |
| **FastAPI** | `5910` | [http://164.52.205.84:5910/docs](https://www.google.com/url?sa=E&source=gmail&q=http://164.52.205.84:5910/docs) | Prediction REST API & Interactive Swagger UI |
| **PostgreSQL** | `5911` | `164.52.205.84:5911` | Metadata backend for Airflow and application store |
| **Prometheus** | `5912` | [http://164.52.205.84:5912](https://www.google.com/url?sa=E&source=gmail&q=http://164.52.205.84:5912) | Metrics collection engine |
| **Grafana** | `5913` | [http://164.52.205.84:5913](https://www.google.com/url?sa=E&source=gmail&q=http://164.52.205.84:5913) | Live monitoring dashboards |
| **Airflow Web** | `6090` | [http://164.52.205.84:6090](https://www.google.com/url?sa=E&source=gmail&q=http://164.52.205.84:6090) | DAG scheduling and pipeline monitoring |
| **MLflow UI** | `6091` | [http://164.52.205.84:6091](https://www.google.com/url?sa=E&source=gmail&q=http://164.52.205.84:6091) | Experiment tracking and model registry |

---

## 🚀 Deployment Strategy

Deployment is completely automated using GitHub Actions (`deploy.yml`):

* **Server Setup (`setup_server`)**: Connects to the host over SSH, creates target directory trees (`/home/da25m591/project/...`), and syncs deployment configs (`docker-compose.yml`, `config/*`).
* **Container Image Builds (`build_*_image`)**: Builds custom Docker images for Airflow and FastAPI, pushing them to GitHub Container Registry (`ghcr.io`).
* **Infrastructure Deployments (`deploy_infra_*`)**: Spins up isolated container instances via Docker Compose for database, MLflow, monitoring, Airflow runtimes, and FastAPI.
* **Code Synchronization (`deploy_code_*`)**: Pushes updated DAGs, ML scripts, and FastAPI code directly to server mounts without full image re-builds.

---

## 🛠 Quickstart (Local Development)

```bash
# Clone repository
git clone [https://github.com/BMoinak98/mlops-project-team23.git](https://github.com/BMoinak98/mlops-project-team23.git)
cd mlops-project-team23

# Start all infrastructure containers
docker compose up -d

# Verify container statuses
docker compose ps
