package com.reviewclassifier.utils

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

class DataLoader(spark: SparkSession) {
  import spark.implicits._

  def loadData(trainPath: String, testPath: String, task: String): (DataFrame, DataFrame) = {
    // Load raw data from Parquet files
    val trainRaw = spark.read.parquet(trainPath)
    val testRaw = spark.read.parquet(testPath)

    // Prepare data based on task
    val (trainProcessed, testProcessed) = task.toLowerCase match {
      case "category" =>
        prepareForCategory(trainRaw, testRaw)
      case "sentiment" =>
        prepareForSentiment(trainRaw, testRaw)
      case _ => throw new IllegalArgumentException(s"Unknown task: $task")
    }

    (trainProcessed, testProcessed)
  }

  private def prepareForCategory(trainDf: DataFrame, testDf: DataFrame): (DataFrame, DataFrame) = {
    val featureCols = trainDf.columns.filter(c => c.startsWith("title_emb_") || c.startsWith("review_emb_")) ++ Array("sentiment_encoded")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("raw_features")

    val trainAssembled = assembler.transform(trainDf)
    val testAssembled = assembler.transform(testDf)

    val scaler = new StandardScaler()
      .setInputCol("raw_features")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)
      .fit(trainAssembled)

    val trainScaled = scaler.transform(trainAssembled)
      .withColumn("label", col("category_encoded").cast("double"))
      .select("features", "label")

    val testScaled = scaler.transform(testAssembled)
      .withColumn("label", col("category_encoded").cast("double"))
      .select("features", "label")

    (trainScaled, testScaled)
  }

  private def prepareForSentiment(trainDf: DataFrame, testDf: DataFrame): (DataFrame, DataFrame) = {
    val featureCols = trainDf.columns.filter(c => c.startsWith("title_emb_") || c.startsWith("review_emb_")) ++ Array("category_encoded")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("raw_features")

    val trainAssembled = assembler.transform(trainDf)
    val testAssembled = assembler.transform(testDf)

    val scaler = new StandardScaler()
      .setInputCol("raw_features")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)
      .fit(trainAssembled)

    val trainScaled = scaler.transform(trainAssembled)
      .withColumn("label", col("sentiment_encoded").cast("double"))
      .select("features", "label")

    val testScaled = scaler.transform(testAssembled)
      .withColumn("label", col("sentiment_encoded").cast("double"))
      .select("features", "label")

    (trainScaled, testScaled)
  }
}
