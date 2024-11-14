FROM apache/airflow:2.9.3-python3.11

# Switch to root to install system dependencies
USER root

# Install Java

RUN mkdir -p /usr/share/man/man1 && apt-get update && apt-get install --no-install-recommends -y \
    openjdk-17-jdk \
    libhdf5-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives

RUN wget https://dlcdn.apache.org/hadoop/common/stable/hadoop-3.4.1.tar.gz && \
    mkdir /opt/hadoop/ && \
    tar -xvzf hadoop-3.4.1.tar.gz -C /opt/hadoop/ && \
    rm hadoop-3.4.1.tar.gz

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    HADOOP_HOME=/opt/hadoop/hadoop-3.4.1 
    
# Switch back to the airflow user
USER airflow

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download necessary JAR files for Delta Lake and MySQL connector
RUN wget -P /home/airflow/.local/lib/python3.11/site-packages/pyspark/jars \
    https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/3.2.0/delta-spark_2.12-3.2.0.jar \
    https://repo1.maven.org/maven2/io/delta/delta-storage/3.2.0/delta-storage-3.2.0.jar 
