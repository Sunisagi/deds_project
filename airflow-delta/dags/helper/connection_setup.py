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
        raise Exception(f"Error setting Hadoop CLASSPATH: {e}")
    return