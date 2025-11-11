package com.reviewclassifier.models
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.{Pipeline, Transformer}
// Importar Regressor
import ml.dmlc.xgboost4j.scala.spark.{
  XGBoostRegressor,
  XGBoostRegressionModel
}
import com.reviewclassifier.Config

class XGBoostModel(config: Config) extends BaseModel {

  override def train(data: DataFrame): Transformer = {
    // La ponderación de clases (class weights) es para clasificación.
    // Se elimina para la regresión.
    
    val Array(trainSet, evalSet) =
      data.randomSplit(Array(0.8, 0.2), seed = config.seed)
    val evalSets = Map("eval" -> evalSet)

    val xgboost = new XGBoostRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      // Eliminado: .setWeightCol("class_weight")
      .setEta(config.eta)
      .setMaxDepth(config.maxDepth)
      // Objetivo de regresión por defecto
      .setObjective("reg:squarederror") 
      // Eliminado: .setNumClass(config.numClasses)
      .setNumRound(config.numRound)
      .setNumWorkers(config.numWorkers)
      .setSeed(config.seed)
      .setEvalSets(evalSets)
      .setTrainTestRatio(0.0) // Usamos nuestra propia división para evalSets

    val pipeline = new Pipeline().setStages(Array(xgboost))
    pipeline.fit(trainSet)
  }

  // Eliminado: private def calculateClassWeights

  override def getName: String = "XGBoostRegressor"
}