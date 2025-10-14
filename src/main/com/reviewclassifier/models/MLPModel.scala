package com.reviewclassifier.models

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.{Pipeline, Transformer}
import com.reviewclassifier.Config

val name : String = "MultilayerPerceptron"

class MLPModel(config: Config) extends BaseModel {

  override def train(data: DataFrame): Transformer = {
    // Get feature dimension from first row
    val featureDim = data.select("features").first()
      .getAs[org.apache.spark.ml.linalg.Vector](0).size

    // Get number of classes
    val numClasses = data.select("label").distinct().count().toInt

    val layers = if (config.modified) {
      // Modified version: Deeper network
      Array(featureDim, 256, 128, 64, numClasses)
    } else {
      // Base version: Simpler network
      Array(featureDim, 64, 32, numClasses)
    }

    val mlp = new MultilayerPerceptronClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setLayers(layers)
      .setMaxIter(config.maxIter)
      .setBlockSize(128)
      .setSeed(42)
      .setStepSize(0.03)

    val pipeline = new Pipeline().setStages(Array(mlp))
    pipeline.fit(data)
  }

  override def getName: String = if (config.modified) "MLP_Modified" else "MLP"
}
