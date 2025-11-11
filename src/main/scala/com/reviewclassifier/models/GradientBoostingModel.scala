package com.reviewclassifier.models

import org.apache.spark.sql.DataFrame
// Importar Regressor
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.{Pipeline, Transformer}
import com.reviewclassifier.Config

class GradientBoostingModel(config : Config) extends BaseModel {
  override def train(data: DataFrame): Transformer = {
    // Usar GBTRegressor
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(config.maxIter)
      .setMaxDepth(config.maxDepth)
      .setStepSize(config.learningRate)
      .setSubsamplingRate(0.7)
      .setMinInstancesPerNode(5)
      .setMaxBins(64)
      .setSeed(config.seed)

    val pipeline = new Pipeline().setStages(Array(gbt))
    pipeline.fit(data)
  }
  override def getName: String = "GradientBoostingRegressor"
}