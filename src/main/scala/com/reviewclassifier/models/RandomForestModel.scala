package com.reviewclassifier.models

import org.apache.spark.sql.DataFrame
// Importar Regressor
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.{Pipeline, Transformer}
import com.reviewclassifier.Config

class RandomForestModel(config: Config) extends BaseModel {

  override def train(data: DataFrame): Transformer = {
    // Usar RandomForestRegressor
    val rf = new RandomForestRegressor()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setNumTrees(config.numTrees)
        .setMaxDepth(config.maxDepth)
        .setMinInstancesPerNode(5)
        .setMaxBins(32)
        .setSubsamplingRate(0.8)
        .setFeatureSubsetStrategy("sqrt")
        .setSeed(config.seed)

    val pipeline = new Pipeline().setStages(Array(rf))
    pipeline.fit(data)
  }

  override def getName: String = "RandomForestRegressor"
}