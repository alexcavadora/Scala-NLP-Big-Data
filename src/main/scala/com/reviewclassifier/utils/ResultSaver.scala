package com.reviewclassifier.utils

import java.io.{File, PrintWriter}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.MLWritable
import scala.util.{Try, Success, Failure}

// 'task' se elimina del constructor
class ResultSaver(resultsDir: String, modelName: String) {

  // 'task' se elimina de la ruta
  private val outputDir = s"$resultsDir/$modelName"

  // Create directory if it doesn't exist
  new File(outputDir).mkdirs()
  new File(s"$outputDir/models").mkdirs()

  // Actualizado para aceptar RegressionMetrics
  def saveMetrics(trainMetrics: RegressionMetrics, testMetrics: RegressionMetrics): Unit = {
    val reportPath = s"$outputDir/regression_report.txt"
    val writer = new PrintWriter(new File(reportPath))

    try {
      writer.write("=" * 60 + "\n")
      writer.write(s"Regression Report: $modelName\n")
      writer.write("=" * 60 + "\n\n")

      writer.write("TRAINING SET METRICS\n")
      writer.write("-" * 60 + "\n")
      writer.write(f"RMSE: ${trainMetrics.rmse}%.6f\n")
      writer.write(f"MSE:  ${trainMetrics.mse}%.6f\n")
      writer.write(f"MAE:  ${trainMetrics.mae}%.6f\n")
      writer.write(f"R2:   ${trainMetrics.r2}%.6f\n\n")

      writer.write("\n\nTEST SET METRICS\n")
      writer.write("-" * 60 + "\n")
      writer.write(f"RMSE: ${testMetrics.rmse}%.6f\n")
      writer.write(f"MSE:  ${testMetrics.mse}%.6f\n")
      writer.write(f"MAE:  ${testMetrics.mae}%.6f\n")
      writer.write(f"R2:   ${testMetrics.r2}%.6f\n\n")

      println(s"Regression report saved to: $reportPath")
    } finally {
      writer.close()
    }

    // Guardar resumen de métricas
    saveMetricsSummary(trainMetrics, testMetrics)
  }

  // Eliminado: saveConfusionMatrix

  private def saveMetricsSummary(trainMetrics: RegressionMetrics, testMetrics: RegressionMetrics): Unit = {
    val path = s"$outputDir/metrics_summary.csv"
    val writer = new PrintWriter(new File(path))
    try {
      writer.write("split,rmse,mse,mae,r2\n")
      writer.write(f"train,${trainMetrics.rmse}%.6f,${trainMetrics.mse}%.6f,${trainMetrics.mae}%.6f,${trainMetrics.r2}%.6f\n")
      writer.write(f"test,${testMetrics.rmse}%.6f,${testMetrics.mse}%.6f,${testMetrics.mae}%.6f,${testMetrics.r2}%.6f\n")
      println(s"Metrics summary saved to: $path")
    } finally {
      writer.close()
    }
  }

  def saveModel(model: Transformer, modelName: String): Unit = {
    val modelPath = s"$outputDir/models/$modelName"
    model match {
      case writable: MLWritable =>
        Try {
          writable.write.overwrite().save(modelPath)
        } match {
          case Success(_) =>
            println(s"Model saved to: $modelPath")
          case Failure(e) =>
            println(s"Warning: Could not save model: ${e.getMessage}")
        }
      case _ =>
        println(s"Warning: Model of type ${model.getClass.getName} is not saveable.")
    }
  }
}