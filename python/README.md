# PySpark-IForest

Pyspark-iforest is a pyspark wrapper for spark-iforest.

Note: it is available for spark version Spark-v2.4.x or later

## Install

Step 1. Package spark-iforest jar and deploy it into spark lib

```bash
cd spark-iforest/

mvn clean package -DskipTests

cp target/spark-iforest-<version>.jar $SPARK_HOME/jars/
```

Step 2. Package pyspark-iforest and install it via pip

```bash
cd spark-iforest/python

python setup.py sdist

pip install dist/pyspark-iforest-<version>.tar.gz
```

## Usage

Parameters are the same with spark-iforest, 
more details can be found in https://github.com/titicaca/spark-iforest#usage

Examples can be found in pyspark_iforest/example/
