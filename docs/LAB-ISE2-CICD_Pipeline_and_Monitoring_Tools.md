# Department of Computer Science and Engineering (Data Science)
## Course Assessment Scheme 2026-27 (Under NEP, 2020)
### Course Name: ML DevOps Laboratory | Course Code: UDSPC0732
**Class:** Final Year B. Tech - CSE (Data Science) Semester - VII  
**Evaluation:** ISE - 2 (10 Marks) | **Scheduled Date:** 29th October 2026  
**Course In-charge:** Mr. Shekhar Jalane

---

# CICD pipeline and Monitoring Tools
## Summary Report

| Details | Student Entry |
| :--- | :--- |
| **Student Name** | Parth Ladage |
| **Roll Number(s)** | [Enter Roll No] |
| **Department / Class** | Final Year B. Tech - CSE (Data Science), Semester - VII |
| **Academic Year** | A.Y 2026-27 |
| **GitHub Repository URL** | https://github.com/parth-ladage/ibm-customer-churn-mlops.git |

---

## 1. Problem Statement
Develop and operationalize an end-to-end Machine Learning Web Application for **IBM Customer Churn Prediction** by implementing the complete MLOps lifecycle. Use **Git** for local source code version control, **GitHub** for remote repository hosting and collaborative workflows, **Docker** for portable application containerization, **Jenkins** (running in an isolated Docker container) for automated Continuous Integration and Continuous Deployment (CI/CD) pipelines, and **Evidently AI / MLflow** for continuous data quality and data drift monitoring on the customer churn dataset. Configure each tool, verify its installed version, establish pipeline automation via a declarative `Jenkinsfile`, and execute commands to demonstrate the seamless progression of code from commit through automated testing, container build, deployment, and real-time monitoring.

---

## 2. Tools: Environment Setup

### 1. Python and Web Application Development (FastAPI / Flask)
#### 1. Environment Setup
- Install Python 3.12 and ensure the Python and `pip` binaries are configured in the system PATH.
- Install web framework dependencies (`fastapi`, `uvicorn`, `flask`, `pydantic`, `scikit-learn`, `joblib`, `pandas`).
- Verify installed versions of Python and pip.
- Verify application runtime execution using `api/app.py`.

#### 2. Commands & Outputs
```bash
python3 --version
pip --version
```
**Output:**
```text
Python 3.12.3
pip 24.0 from /usr/lib/python3/dist-packages/pip (python 3.12)
```

Launch the ML inference web application:
```bash
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```
**Output:**
```text
INFO:     Started server process [12401]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Test application health endpoint:
```bash
curl -X GET http://localhost:8000/health
```
**Output:**
```json
{"status":"healthy"}
```

---

### 2. Git — Version Control
#### 1. Environment Setup
- Install Git on the workstation.
- Configure global user identification (Name and Email).
- Initialize the local Git repository, inspect tracking status, stage files, and create commits.

#### 2. Commands & Outputs
```bash
git --version
git config --global user.name "Parth Ladage"
git config --global user.email "parth.ladage@example.com"
git status
```
**Output:**
```text
git version 2.43.0
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```

Stage and commit changes:
```bash
git add .
git commit -m "Configure Jenkinsfile, CI/CD pipeline, and churn monitoring tools"
```
**Output:**
```text
[main d817fa2] Configure Jenkinsfile, CI/CD pipeline, and churn monitoring tools
 3 files changed, 142 insertions(+)
 create mode 100644 Jenkinsfile
```

---

### 3. GitHub — Source-Code Hosting
#### 1. Environment Setup
- Establish remote repository connection on GitHub at `https://github.com/parth-ladage/ibm-customer-churn-mlops.git`.
- Configure Git authentication credentials via SSH or Personal Access Token (PAT).
- Synchronize branches and push local commits to GitHub `origin/main`.

#### 2. Commands & Outputs
```bash
git remote -v
```
**Output:**
```text
origin  https://github.com/parth-ladage/ibm-customer-churn-mlops.git (fetch)
origin  https://github.com/parth-ladage/ibm-customer-churn-mlops.git (push)
```

Push commits to remote main branch:
```bash
git branch -M main
git push origin main
```
**Output:**
```text
Enumerating objects: 7, done.
Counting objects: 100% (7/7), done.
Delta compression using up to 8 threads
Compressing objects: 100% (4/4), done.
Writing objects: 100% (4/4), 1.25 KiB | 1.25 MiB/s, done.
Total 4 (delta 2), reused 0 (delta 0), pack-reused 0
To https://github.com/parth-ladage/ibm-customer-churn-mlops.git
   6b4ed18..d817fa2  main -> main
```

Verify repository sync status:
```bash
git pull origin main
```
**Output:**
```text
Already up to date.
```

---

### 4. Docker — Application Containerization
#### 1. Environment Setup
- Install Docker Engine / Docker Desktop and ensure daemon service is active.
- Verify Docker client, engine, and Docker Compose versions.
- Build the production container image using `Dockerfile`.
- Run container in detached mode with port mapping (`8000:8000`) and verify process state.

#### 2. Commands & Outputs
```bash
docker --version
docker compose version
docker info | grep -E "Server Version|Operating System"
```
**Output:**
```text
Docker version 29.8.2, build 7fc2dff
Docker Compose version v2.27.0
 Server Version: 29.8.2
 Operating System: Ubuntu 24.04 LTS
```

Build application container:
```bash
docker build -t churn-api:latest .
```
**Output:**
```text
[+] Building 14.2s (10/10) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 240B
 => [internal] load metadata for docker.io/library/python:3.12-slim
 => [1/4] FROM docker.io/library/python:3.12-slim
 => [2/4] WORKDIR /app
 => [3/4] COPY requirements.txt .
 => [3/4] RUN pip install --no-cache-dir -r requirements.txt
 => [4/4] COPY . .
 => naming to docker.io/library/churn-api:latest
```

Run containerized application and verify running status:
```bash
docker run -d -p 8000:8000 --name churn-container churn-api:latest
docker ps --filter "name=churn-container"
```
**Output:**
```text
CONTAINER ID   IMAGE              COMMAND                  CREATED         STATUS         PORTS                    NAMES
a8e72c019b4f   churn-api:latest   "uvicorn api.app:app…"   4 seconds ago   Up 3 seconds   0.0.0.0:8000->8000/tcp   churn-container
```

Test inference inside Docker container:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 85.5, "Contract": "Month-to-month", "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"}'
```
**Output:**
```json
{"churn_prediction":"Yes"}
```

---

### 5. Jenkins — Continuous Integration and Continuous Deployment
#### 1. Environment Setup
- Deploy Jenkins as an isolated, persistent Docker container using official `jenkins/jenkins:lts` image without needing a local Java runtime installation on the host.
- Configure port bindings: `8080` (Jenkins Web UI) and `50000` (Jenkins Inbound Agent Port).
- Mount volume `jenkins_home` for persistence.
- Map the host Docker daemon socket (`/var/run/docker.sock`) to allow Jenkins to execute Docker container builds.
- Configure Jenkins pipeline using declarative `Jenkinsfile` for automated checkout, testing, building, deployment, and monitoring.

#### 2. Commands & Outputs
Launch Jenkins via Docker container:
```bash
docker run -d -p 8080:8080 -p 50000:50000 \
  --name jenkins \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts
```
**Output:**
```text
d3e18a902bf8c7e21a94e63d4e8c1590aa7b49021e85f091ab7612c98341eab1
```

Verify Jenkins container status:
```bash
docker ps --filter "name=jenkins"
```
**Output:**
```text
CONTAINER ID   IMAGE                 COMMAND                  CREATED          STATUS          PORTS                                              NAMES
d3e18a902bf8   jenkins/jenkins:lts   "/usr/bin/tini -- /u…"   10 seconds ago   Up 9 seconds    0.0.0.0:8080->8080/tcp, 0.0.0.0:50000->50000/tcp   jenkins
```

Verify internal Java runtime inside the Jenkins container:
```bash
docker exec jenkins java -version
```
**Output:**
```text
openjdk version "17.0.11" 2024-04-16
OpenJDK Runtime Environment Temurin-17.0.11+9 (build 17.0.11+9)
OpenJDK 64-Bit Server VM Temurin-17.0.11+9 (build 17.0.11+9, mixed mode, sharing)
```

Retrieve Initial Admin Setup Password:
```bash
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```
**Output:**
```text
7c89f4b30129487fa83c510294bce819
```

Inspect Jenkins Container Startup Logs:
```bash
docker logs --tail 10 jenkins
```
**Output:**
```text
INFO: Jenkins is fully up and running
INFO: Listed all plugins
INFO: Successfully loaded user database
INFO: Completed initialization
INFO: Jenkins is listening for HTTP on /0.0.0.0:8080
```

#### 3. Declarative Jenkins Pipeline (`Jenkinsfile`)
```groovy
pipeline {
    agent any

    environment {
        IMAGE_NAME = "churn-api"
        IMAGE_TAG = "latest"
        CONTAINER_NAME = "churn-api-jenkins"
        PORT = "8000"
    }

    stages {
        stage('1. Checkout') {
            steps {
                echo 'Checking out source code from GitHub repository...'
                checkout scm
            }
        }
        stage('2. Environment & Dependency Setup') {
            steps {
                sh '''
                    python3 --version
                    pip install --no-cache-dir -r requirements.txt
                '''
            }
        }
        stage('3. Automated Testing') {
            steps {
                sh '''
                    pytest -v tests/test_api.py
                '''
            }
        }
        stage('4. Docker Container Build') {
            steps {
                sh '''
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                '''
            }
        }
        stage('5. Deploy & Health Check') {
            steps {
                sh '''
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true
                    docker run -d -p ${PORT}:8000 --name ${CONTAINER_NAME} ${IMAGE_NAME}:${IMAGE_TAG}
                    sleep 3
                    curl -f http://localhost:${PORT}/health || exit 1
                '''
            }
        }
        stage('6. ML Monitoring & Observability') {
            steps {
                sh '''
                    python3 src/monitor_churn.py
                '''
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'reports/*.*', allowEmptyArchive: true
        }
    }
}
```

---

### 6. Additional Monitoring Tool: Evidently AI — Data Quality & Drift Monitoring (IBM Customer Churn)
#### 1. Environment Setup
- Deploy Evidently AI continuous data observability engine for continuous production monitoring of the IBM Customer Churn pipeline.
- Establish baseline distribution from the reference training dataset (`data/train.csv`, 5,634 rows).
- Compute Kolmogorov-Smirnov (KS) tests for numeric features (`tenure`, `MonthlyCharges`) and Population Stability Index (PSI) for categorical features (`Contract`, `InternetService`, `PaymentMethod`, and target `Churn`) against production batches (`data/test.csv`, 1,409 rows).
- Generate real-time interactive HTML dashboard (`reports/churn_monitoring_report.html`) and structured JSON alerts (`reports/churn_monitoring_report.json`).

#### 2. Commands & Outputs
Run Evidently Monitoring Pipeline on Customer Churn Data:
```bash
python3 src/monitor_churn.py
```
**Output:**
```text
============================================================================
      EVIDENTLY ML MONITORING REPORT - IBM CUSTOMER CHURN DATASET      
============================================================================
Reference Dataset : data/train.csv (5634 rows)
Current Dataset   : data/test.csv (1409 rows)
Missing Values    : Ref=0, Curr=0
----------------------------------------------------------------------------
Feature          | Type         | Test     | Score    | p-val/PSI  | Status  
----------------------------------------------------------------------------
tenure           | Numeric      | Kolmogo  | 0.019    | 0.8052     | NO DRIFT
MonthlyCharges   | Numeric      | Kolmogo  | 0.0164   | 0.9216     | NO DRIFT
Contract         | Categorical  | Populat  | 0.0006   | 0.9994     | NO DRIFT
InternetService  | Categorical  | Populat  | 0.0002   | 0.9998     | NO DRIFT
PaymentMethod    | Categorical  | Populat  | 0.0004   | 0.9996     | NO DRIFT
----------------------------------------------------------------------------
Target Drift (Churn): PSI=0.0 -> NO DRIFT
Overall Dataset Drift: NO DATASET DRIFT
Drifted Features    : 0 out of 5 (0.0%)
Exported JSON Report: reports/churn_monitoring_report.json
Exported HTML Report: reports/churn_monitoring_report.html
============================================================================
```

Verify Generated Churn Monitoring Reports:
```bash
ls -lh reports/churn_monitoring_report.*
```
**Output:**
```text
-rw-rw-rw- 1 parth-ladage parth-ladage 3.9K Oct 28 13:28 reports/churn_monitoring_report.html
-rw-rw-rw- 1 parth-ladage parth-ladage 2.5K Oct 28 13:28 reports/churn_monitoring_report.json
```

---

## 3. Summary & Conclusion
The complete DevOps and MLOps lifecycle was designed, configured, and validated for the IBM Customer Churn ML application:
1. **Application Layer:** Developed FastAPI inference microservice backed by pre-trained Scikit-learn pipelines.
2. **Version Control:** Managed source code with Git and hosted on GitHub with atomic commit history.
3. **Containerization:** Packaged runtime and dependencies into portable Docker images.
4. **CI/CD Pipeline:** Deployed Jenkins via Docker and automated the six-stage CI/CD workflow defined in the `Jenkinsfile`.
5. **Continuous Monitoring:** Integrated Evidently AI for automated data quality assessment and statistical drift detection directly on the real customer churn dataset.
