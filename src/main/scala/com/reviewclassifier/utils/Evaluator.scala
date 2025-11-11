package com.reviewclassifier.utils

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.evaluation.RegressionEvaluator

case class RegressionMetrics(
  rmse: Double,
  mse: Double,
  mae: Double,
  r2: Double,
  predictions: DataFrame
)

class Evaluator(spark: SparkSession) {
  import spark.implicits._

  def evaluate(model: Transformer, data: DataFrame, split: String): RegressionMetrics = {
    // Make predictions
    val predictions = model.transform(data)

    // RMSE
    val rmseEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = rmseEvaluator.evaluate(predictions)

    // MSE
    val mseEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("mse")
    val mse = mseEvaluator.evaluate(predictions)

    // MAE
    val maeEvaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("mae")
    val mae = maeEvaluator.evaluate(predictions)

    // R-squared (R2)
    val r2Evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("r2")
    val r2 = r2Evaluator.evaluate(predictions)


    println(s"\n=== $split Metrics ===")
    println(f"RMSE: $rmse%.4f")
    println(f"MSE: $mse%.4f")
    println(f"MAE: $mae%.4f")
    println(f"R2: $r2%.4f")

    RegressionMetrics(rmse, mse, mae, r2, predictions)
  }
}