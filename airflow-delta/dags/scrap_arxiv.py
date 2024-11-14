from datetime import datetime
from airflow.decorators import dag, task
import requests
import xmltodict
import json
from pyarrow import fs
import os
import subprocess

def set_classpath() :
    try:
        hadoop_home = os.environ.get("HADOOP_HOME")
        if not hadoop_home:
            raise EnvironmentError("HADOOP_HOME environment variable is not set")

        hdfs_classpath = subprocess.check_output([f"{hadoop_home}/bin/hdfs", "classpath", "--glob"]).decode().strip()
        
        # Set the CLASSPATH environment variable
        os.environ["CLASSPATH"] = hdfs_classpath
        print("Hadoop CLASSPATH set successfully.")
    except Exception as e:
        raise f"Error setting Hadoop CLASSPATH: {e}"
    return
@dag(
    schedule=None,
    start_date=datetime(2024, 11, 13),
    catchup=False
)
def scraping_arxiv() :  
      
    @task()
    def query_arxiv() :
        
        year = "2024"
        start = 0
        max_results = 1000
        query = f'submittedDate:[{year}01010000 TO {year}12312359]'
        url = f'http://export.arxiv.org/api/query?search_query={query}&start={start}&max_results={max_results}'
        response = requests.get(url)
        if response.status_code == 200:
            json_data = json.dumps(xmltodict.parse(response.content), indent=4) 
            return json_data
        else:
            raise f"Error: Received status code {response.status_code} from arXiv API."      

    @task()
    def write_json(json_data) :
        
        # Get the Hadoop classpath using `hdfs classpath --glob`
        
        set_classpath()
        hdfs = fs.HadoopFileSystem("namenode", 8020)

        # hdfs = HDFileSystem("namenode", 8020) 

        with hdfs.open_output_stream('json/arxiv.json') as file :
            file.write(json_data.encode('utf-8'))

    json_data = query_arxiv()
    write_json(json_data)

scraping_arxiv()