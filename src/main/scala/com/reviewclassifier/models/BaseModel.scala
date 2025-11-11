package com.reviewclassifier.models

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{PipelineModel, Transformer}
import com.reviewclassifier.Config

trait BaseModel {
  def train(data: DataFrame): Transformer
  def getName: String
}

object ModelFactory {
  def getModel(modelName: String, config: Config): BaseModel = {
    modelName.toLowerCase match {
        case "randomforest" | "rf" => new RandomForestModel(config)
        // 'logisticregression' ahora apunta a 'LinearRegressionModel'
        case "linearregression" | "lr" => new LinearRegressionModel(config)
        case "gradientboosting" | "gbt" => new GradientBoostingModel(config)
        case "xgboost" => new XGBoostModel(config)
        
        // Eliminados: naivebayes, mlp, logisticregression (nombre antiguo)
        case _ => throw new IllegalArgumentException(s"Unknown regression model: $modelName")
    }
  }
}
