# app/Dockerfile

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    openjdk-17-jdk \
    libhdf5-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives

RUN wget https://dlcdn.apache.org/hadoop/common/stable/hadoop-3.4.1.tar.gz && \
mkdir /opt/hadoop/ && \
tar -xvzf hadoop-3.4.1.tar.gz -C /opt/hadoop/ && \
rm hadoop-3.4.1.tar.gz

RUN mkdir /opt/src/

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    HADOOP_HOME=/opt/hadoop/hadoop-3.4.1 

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

