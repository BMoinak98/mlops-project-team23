# Self-Hosted Airflow & Monitoring Pipeline

An automated data pipeline infrastructure running Apache Airflow, PostgreSQL, Prometheus, and Grafana. Deployed automatically to a remote host via GitHub Actions using Docker Compose.

---

## 🏗️ Architecture

```
[ Developer Push ] ──► [ GitHub Repo ]
                            │
                            ├──► [ GitHub Actions ]
                            │          │
                            │          ├──► Builds App Image ──► [ GHCR.io ]
                            │          │                              │
                            │     (SCP & SSH)                         │
                            ▼          ▼                              │
             ┌─────────────────────────────────────────────────┐      │
             │            Target Server (Docker Host)          │      │
             │                                                 │      │
             │  ┌─────────────────┐     ┌──────────────────┐   │      │
             │  │ Airflow Web/Sch │ ──► │ PostgreSQL DB    │   │      │
             │  └────────┬────────┘     └──────────────────┘   │      │
             │           │                                     │      │
             │     (DockerOperator)                            │      │
             │           │ (Pulls & Runs Container)            │      │
             │           ▼                                     ▼      │
             │  ┌─────────────────┐     ┌──────────────────┐   │      │
             │  │ Custom App Code │ ◄───┴──────────────────┼───┘      │
             │  └─────────────────┘                        │          │
             │                          ┌──────────────────┐          │
             │                          │ Prometheus/Graf. │          │
             │                          └──────────────────┘          │
             └─────────────────────────────────────────────────┘
```

---

## 🛠️ Stack & Services

| Service | Port | Description |
| :--- | :--- | :--- |
| **Airflow Webserver** | `8080` | Pipeline orchestration UI |
| **PostgreSQL** | Internal | Airflow metadata database |
| **Prometheus** | `9090` | Metrics scraping & monitoring |
| **Grafana** | `3000` | Analytics & monitoring dashboard |
| **Custom App Image** | Internal | Python worker image pulled dynamically from GHCR |

---

## 🚀 One-Time Server Setup

Execute these commands **once** on your target server before running GitHub Actions:

### 1. Install Docker & Docker Compose
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
```

### 2. Configure Permissions
Grant your SSH deployment user access to run Docker without `sudo`:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 3. Prepare Target Directory
```bash
sudo mkdir -p /opt/my-pipeline
sudo chown -R $USER:$USER /opt/my-pipeline
```

---

## 🔐 Required GitHub Secrets

Add the following credentials in your GitHub Repository under **Settings > Secrets and variables > Actions**:

*   `SSH_HOST` — Public IP or domain name of your server.
*   `SSH_USER` — Username used for SSH access.
*   `SSH_PRIVATE_KEY` — Private key corresponding to `~/.ssh/authorized_keys` on the server.

---

## 🔄 Deployment Process (CI/CD)

Every time code is pushed to the `main` branch:

1. **Build & Package:** GitHub Actions builds `app/Dockerfile` and pushes the resulting image to `ghcr.io`.
2. **File Sync:** Configuration files (`docker-compose.yml`, `prometheus.yml`) and DAG files (`dags/`) are copied to `/opt/my-pipeline/` on the server via SCP.
3. **Container Launch:** GitHub Actions executes `docker compose up -d` over SSH to update all running services.
4. **Execution:** Airflow detects the DAG file and uses the server's Docker engine to pull and execute the application container from GHCR.

---

## 🌐 Default Access URLs

*   **Airflow Webserver:** `http://<YOUR_SERVER_IP>:8080`
*   **Grafana Dashboard:** `http://<YOUR_SERVER_IP>:3000` *(Default Login: `admin` / `admin`)*
*   **Prometheus Metrics:** `http://<YOUR_SERVER_IP>:9090`