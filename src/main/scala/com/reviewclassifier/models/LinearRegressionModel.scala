// Renombrar archivo a LinearRegressionModel.scala
package com.reviewclassifier.models

import org.apache.spark.sql.DataFrame
// Importar LinearRegression
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, Transformer}
import com.reviewclassifier.Config

// Renombrar clase
class LinearRegressionModel(config: Config) extends BaseModel {
  override def train(data: DataFrame): Transformer = {
    // Usar LinearRegression
    val lr = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(config.maxIter * 2)
      .setRegParam(config.regParam * 0.75)
      .setElasticNetParam(config.elasticNetParam)
      .setTol(1e-6)
      .setFitIntercept(false)
      // Eliminado: .setFamily("multinomial")

    val pipeline = new Pipeline().setStages(Array(lr))
    pipeline.fit(data)
  }

  override def getName: String = "LinearRegression"
}