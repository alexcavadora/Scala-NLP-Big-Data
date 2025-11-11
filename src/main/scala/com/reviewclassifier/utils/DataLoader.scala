package com.reviewclassifier.utils

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

class DataLoader(spark: SparkSession) {
  import spark.implicits._

  // 'task' se reemplaza por 'labelCol'
  def loadData(trainPath: String, testPath: String, labelCol: String): (DataFrame, DataFrame) = {
    // Load raw data from Parquet files
    val trainRaw = spark.read.parquet(trainPath)
    val testRaw = spark.read.parquet(testPath)

    // Validar que la columna de etiqueta exista
    if (!trainRaw.columns.contains(labelCol)) {
        throw new IllegalArgumentException(s"Label column '$labelCol' not found in training data.")
    }
    
    // Preparar datos para regresión
    val (trainProcessed, testProcessed) = prepareData(trainRaw, testRaw, labelCol)

    (trainProcessed, testProcessed)
  }

  // Métodos 'prepareForCategory' y 'prepareForSentiment' eliminados
  // Reemplazados por un 'prepareData' genérico

  private def prepareData(trainDf: DataFrame, testDf: DataFrame, labelCol: String): (DataFrame, DataFrame) = {
    // Asume que todas las columnas 'emb' son características
    // Excluye la columna de etiqueta de las características

    val featureCols = trainDf.columns.filter(c => c.startsWith("desc_emb_"))

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

    // Usa 'labelCol' para crear la columna 'label'
    val trainScaled = scaler.transform(trainAssembled)
      .withColumn("label", col(labelCol).cast("double"))
      .select("features", "label")

    val testScaled = scaler.transform(testAssembled)
      .withColumn("label", col(labelCol).cast("double"))
      .select("features", "label")

    (trainScaled, testScaled)
  }
}