FROM apache/airflow:2.9.3-python3.11

# Switch to root to install system dependencies
USER root

# Install Java and MySQL client libraries
RUN mkdir -p /usr/share/man/man1 && apt-get update && apt-get install --no-install-recommends -y \
    openjdk-17-jdk \
    libmysqlclient-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives

# Switch back to the airflow user
USER airflow

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download necessary JAR files for Delta Lake and MySQL connector
RUN wget -P /home/airflow/.local/lib/python3.11/site-packages/pyspark/jars \
    https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/3.2.0/delta-spark_2.12-3.2.0.jar \
    https://repo1.maven.org/maven2/io/delta/delta-storage/3.2.0/delta-storage-3.2.0.jar \
    https://repo1.maven.org/maven2/com/mysql/mysql-connector-j/9.0.0/mysql-connector-j-9.0.0.jar
