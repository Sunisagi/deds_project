from datetime import datetime
from airflow.decorators import dag, task
import requests
import xmltodict
import json
from pyarrow import fs
from helper.connection_setup import set_classpath

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
            raise ConnectionError(f"Error: Received status code {response.status_code} from arXiv API.")      

    @task()
    def write_json(json_data) :
        
        # Get the Hadoop classpath using `hdfs classpath --glob`
        
        set_classpath()
        hdfs = fs.HadoopFileSystem("namenode", 8020)

        # hdfs = HDFileSystem("namenode", 8020) 

        with hdfs.open_output_stream('/scraping/arxiv.json') as file :
            file.write(json_data.encode('utf-8'))

    json_data = query_arxiv()
    write_json(json_data)

scraping_arxiv()