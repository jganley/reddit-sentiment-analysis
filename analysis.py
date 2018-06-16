from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os

def write_parquet(context):
    # Read from JSON
    comments = sqlContext.read.json("superbowl_comments.json")

    # Write the Parquets
    comments.write.parquet("superbowl_comments.parquet")

def main(context):

    # Check if the parquet file has already been written
    if not os.path.exists("superbowl_comments.parquet"):
        write_parquet(context)

    # Read the parquets
    comments = sqlContext.read.parquet("superbowl_comments.parquet")
    comments.registerTempTable("commentsTable")

    comments.show()

if __name__ == "__main__":
    conf = SparkConf().setAppName("SuperBowl Sentiment Analysis")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    import cleantext
    main(sqlContext)
