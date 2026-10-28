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
                echo 'Setting up Python runtime and installing dependencies...'
                sh '''
                    python3 --version
                    pip --version
                    pip install --no-cache-dir -r requirements.txt
                '''
            }
        }

        stage('3. Automated Testing') {
            steps {
                echo 'Running automated tests with Pytest...'
                sh '''
                    pytest -v tests/test_api.py
                    pytest -v tests/test_iris.py
                '''
            }
        }

        stage('4. Docker Container Build') {
            steps {
                echo "Building Docker container image: ${IMAGE_NAME}:${IMAGE_TAG}..."
                sh '''
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                    docker images ${IMAGE_NAME}:${IMAGE_TAG}
                '''
            }
        }

        stage('5. Deploy & Health Check') {
            steps {
                echo "Deploying containerized application on port ${PORT}..."
                sh '''
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true
                    docker run -d -p ${PORT}:8000 --name ${CONTAINER_NAME} ${IMAGE_NAME}:${IMAGE_TAG}
                    sleep 3
                    curl -f http://localhost:${PORT}/health || exit 1
                    echo "Application deployed and healthy!"
                '''
            }
        }

        stage('6. ML Monitoring & Observability') {
            steps {
                echo 'Executing Evidently ML monitoring and data drift checks...'
                sh '''
                    python3 src/monitor_iris.py
                '''
            }
        }
    }

    post {
        always {
            echo 'Archiving test and monitoring artifacts...'
            archiveArtifacts artifacts: 'reports/*.*', allowEmptyArchive: true
        }
        success {
            echo 'Pipeline completed successfully: All stages passed quality gates!'
        }
        failure {
            echo 'Pipeline failed: Please review stage logs for details.'
        }
    }
}

