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

def write_parquet(sqlContext):
    # Read from JSON
    comments = sqlContext.read.json("superbowl_comments.json")

    # Write the Parquets
    comments.write.parquet("superbowl_comments.parquet")

def create_dataframe(sqlContext, modelDataframe):
    def parse_text(text):
        return cleantext.sanitize(text)

    parse_udf = udf(parse_text, ArrayType(StringType()))
    modelDataframe = modelDataframe.withColumn("udf_results", parse_udf(col("body")))

    return modelDataframe

def train_cv_model(modelDataframe):
    cv = CountVectorizer(inputCol="udf_results", outputCol="features", binary=True, minDF=5.0)
    model = cv.fit(modelDataframe)
    model.write().overwrite().save("models/cvModel")

def transform_model(sqlContext, modelDataframe):
    # Load the CV model
    model = CountVectorizerModel.load("models/cvModel")
    # Transform the data frame
    transformedDf = model.transform(modelDataframe)

    return transformedDf

def create_models(sqlContext, modelDataframe):
    modelDataframe.registerTempTable("modelDataframeTable")

    # Create dataframes to use on the positive and negative models
    pos = sqlContext.sql("SELECT pos_label AS label, features FROM modelDataframeTable")
    neg = sqlContext.sql("SELECT neg_label AS label, features FROM modelDataframeTable")

    # Initialize two logistic regression models.
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).setThreshold(0.2)
    neglr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).setThreshold(0.25)
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator()
    negEvaluator = BinaryClassificationEvaluator()
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=2)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=2)
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.write().overwrite().save("models/posModel")
    negModel.write().overwrite().save("models/negModel")

def create_fullDataframe(sqlContext, comments):
    comments.registerTempTable("commentsTable")
    # Create dataframe with all the data
    fullDataframe = sqlContext.sql("SELECT commentsTable.comments_id AS id, commentsTable.comments_body AS body, commentsTable.comments_author AS author, commentsTable.comments_created_utc AS created_utc, commentsTable.comments_subreddit_id AS subreddit_id, commentsTable.comments_link_id AS link_id, commentsTable.comments_parent_id AS parent_id, commentsTable.comments_score AS score, commentsTable.comments_controversiality AS controversiality, commentsTable.comments_gilded AS gilded FROM commentsTable")
    fullDataframe = create_dataframe(sqlContext, fullDataframe)

    # Transform the full data
    fullDataframe = transform_model(sqlContext, fullDataframe)

    fullDataframe.write.parquet("fullDataframe.parquet")

def main(sqlContext):

    # Check if the parquet file has already been written
    if not os.path.exists("superbowl_comments.parquet"):
        write_parquet(sqlContext)

    # Read the parquets
    comments = sqlContext.read.parquet("superbowl_comments.parquet")
    comments.registerTempTable("commentsTable")

    # Read the labels csv
    labels = sqlContext.read.format('csv').options(header='true', inferSchema='true').load("labels.csv")
    labels.registerTempTable("labelsTable")

    # Create Dataframe to Train the Model
    modelDataframe = sqlContext.sql("SELECT commentsTable.comments_id AS id, commentsTable.comments_body AS body, commentsTable.comments_author AS author, commentsTable.comments_created_utc AS created_utc, commentsTable.comments_subreddit_id AS subreddit_id, commentsTable.comments_link_id AS link_id, commentsTable.comments_parent_id AS parent_id, commentsTable.comments_score AS score, commentsTable.comments_controversiality AS controversiality, commentsTable.comments_gilded AS gilded FROM commentsTable INNER JOIN labelsTable ON commentsTable.comments_id = labelsTable.label_id")
    modelDataframe = create_dataframe(sqlContext, modelDataframe)

    # Fit the CountVectorizer model
    if(not os.path.exists("models/cvModel")):
        train_cv_model(modelDataframe)

    # Use model to transform the data
    modelDataframe = transform_model(sqlContext, modelDataframe)
    modelDataframe.registerTempTable("modelDataframeTable")
    modelDataframe = sqlContext.sql("SELECT modelDataframeTable.*, IF(labelsTable.label=1, 1, 0) AS pos_label, IF(labelsTable.label=-1, 1, 0) AS neg_label FROM modelDataframeTable INNER JOIN labelsTable ON modelDataframeTable.id = labelsTable.label_id")

    if(not os.path.exists("models/negModel") or not os.path.exists("models/posModel")):
        create_models(sqlContext, modelDataframe)

    # Load the positive and negative models back in
    posModel = CrossValidatorModel.load("models/posModel")
    negModel = CrossValidatorModel.load("models/negModel")

    if(not os.path.exists("fullDataframe.parquet")):
        create_fullDataframe(sqlContext, comments)

    # Load the full dataframe back in
    fullDataframe = sqlContext.read.parquet("fullDataframe.parquet")
    fullDataframe.registerTempTable("fullDataframeTable")

    # Get rid of comments that are sarcastic or removed
    fullDataframe = sqlContext.sql("SELECT * FROM fullDataframeTable WHERE fullDataframeTable.body NOT LIKE '%/s%' AND fullDataframeTable.body NOT LIKE '&gt%' AND fullDataframeTable.body NOT LIKE '%[removed]%'")




if __name__ == "__main__":
    conf = SparkConf().setAppName("SuperBowl Sentiment Analysis")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    import cleantext
    main(sqlContext)
