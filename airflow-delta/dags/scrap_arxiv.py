from datetime import datetime
import time
from airflow.decorators import dag, task
import requests
import xmltodict
import json
from pyarrow import fs
from helper.connection_setup import set_classpath
from airflow.models import Variable
from airflow.providers.http.operators.http import SimpleHttpOperator

max_results = 1000

@dag(
    schedule='@daily',
    start_date=datetime(2024, 11, 13),
    catchup=False
)
def scraping_arxiv() :  
      
    @task()
    def query_arxiv() :      

        try:
            vars = Variable.get("scraping_config", deserialize_json=True)            
        except KeyError:
            print('No cache config, use offset=0, year=2024, pages=1000')
            vars = {
                'year': '2024',
                'start': 0,
                'max_results': 1000,
            }
            Variable.set(key="scraping_config", value=vars, serialize_json=True) 
        year = vars['year']
        start = int(vars['start'])
        max_results = int(vars['max_results'])
        max_retries = 5
        num_retries = 1
        results = []

        query = f'submittedDate:[{year}01010000 TO {year}12312359]'
        while (max_results > 0) and (num_retries <= max_retries) :
            url = f'http://export.arxiv.org/api/query?search_query={query}&start={start}&max_results={max_results}'
            print(f'retry number: {num_retries}')
            response = requests.get(url)
            if response.status_code == 200:
                dict_data = xmltodict.parse(response.content)
                if 'entry' in dict_data['feed'] :
                    num_results = len(dict_data['feed']['entry'])
                    print(f'query: {num_results}')
                    max_results -= num_results
                    start += num_results
                    results.append(dict_data)
                else :
                    time.sleep(20)
                    print('No data return')
                num_retries += 1
            else:
                raise ConnectionError(f"Error: Received status code {response.status_code} from arXiv API.") 
        base_json = results.pop(0)   
        for x in results :
            base_json['feed']['entry'] += x['feed']['entry']  
        json_data = json.dumps(base_json, indent=4)  
        print('scraping success')          
        return json_data 

    @task()
    def write_json(json_data) :
        
        # Get the Hadoop classpath using `hdfs classpath --glob`
        
        set_classpath()
        hdfs = fs.HadoopFileSystem("namenode", 8020)

        with hdfs.open_output_stream(f'/scraping/arxiv_{datetime.now().date()}.json') as file :
            file.write(json_data.encode('utf-8'))
        
    @task()
    def update_offset() :
        try:
            vars = Variable.get("scraping_config", deserialize_json=True)
            vars['start'] = int(vars['start']) + int(vars['max_results']) 
        except KeyError:
            vars = {
                'year': '2024',
                'start': 0,
                'max_results': 1000,
            }
        Variable.set(key="scraping_config", value=vars, serialize_json=True) 

    trigger_retrain = SimpleHttpOperator(
        task_id='trigger_retrain',
        http_conn_id='ml_connection',
        endpoint='/retrain',
        method='POST',
        headers={"Content-Type": "application/json"}
    )
        
    json_data = query_arxiv()
    write_json(json_data) >> update_offset() >> trigger_retrain  

scraping_arxiv()