package com.reviewclassifier.models
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import ml.dmlc.xgboost4j.scala.spark.{
  XGBoostClassifier,
  XGBoostClassificationModel
}
import com.reviewclassifier.Config

class XGBoostModel(config: Config) extends BaseModel {

  override def train(data: DataFrame): Transformer = {
    val classWeights = calculateClassWeights(data)
    val labelToWeightUDF =
      udf((label: Double) => classWeights.getOrElse(label, 1.0))
    val dataWithWeights =
      data.withColumn("class_weight", labelToWeightUDF(col("label")))
    val trainingData =
      dataWithWeights.select("features", "label", "class_weight")
    val Array(trainSet, evalSet) =
      trainingData.randomSplit(Array(0.8, 0.2), seed = config.seed)
    val evalSets = Map("eval" -> evalSet)

    val xgboost = new XGBoostClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setWeightCol("class_weight")
      .setEta(config.eta)
      .setMaxDepth(config.maxDepth)
      .setObjective(config.objective)
      .setNumClass(config.numClasses)
      .setNumRound(config.numRound)
      .setNumWorkers(config.numWorkers)
      .setSeed(config.seed)
      .setEvalSets(evalSets)
      .setTrainTestRatio(0.0)

    val pipeline = new Pipeline().setStages(Array(xgboost))
    pipeline.fit(trainSet)
    // xgboost.fit(trainingData)
  }

  private def calculateClassWeights(data: DataFrame): Map[Double, Double] = {
    val classCounts = data.groupBy("label").count().collect()
    val totalSamples = data.count().toDouble
    val numClasses = classCounts.length.toDouble

    classCounts.map { row =>
      val label = row.getAs[Double]("label")
      val count = row.getAs[Long]("count").toDouble
      val weight = totalSamples / (numClasses * count)
      (label, weight)
    }.toMap
  }

  override def getName: String = "XGBoost"
}
