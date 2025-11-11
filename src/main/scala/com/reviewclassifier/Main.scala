package com.reviewclassifier

import org.apache.spark.sql.SparkSession
import scopt.OParser
import com.reviewclassifier.models._
import com.reviewclassifier.utils._

case class Config(
    model: String = "RandomForest",
    labelCol: String = "score", // Columna de etiqueta para regresión
    trainPath: String = "data/train.parquet",
    testPath: String = "data/test.parquet",
    resultsDir: String = "results",
    maxIter: Int = 100,
    maxDepth: Int = 8,
    numTrees: Int = 100, // Reducido para regresión, ajustar según sea necesario
    learningRate: Double = 0.1,
    regParam: Double = 0.01,
    elasticNetParam: Double = 0.8,
    seed: Int = 64,
    eta: Double = 0.1,
    numRound: Int = 100,
    numWorkers: Int = 4
    // Eliminados: task, numClasses, objective
)

object Main {
  def main(args: Array[String]): Unit = {
    val builder = OParser.builder[Config]
    val parser = {
      import builder._
      OParser.sequence(
        programName("ReviewRegressor"),
        head("Review Regressor", "0.1.0"),
        opt[String]('m', "model")
          .action((x, c) => c.copy(model = x))
          .text(
            "Model: RandomForest, LinearRegression, GradientBoosting, XGBoost"
          ),
        opt[String]("label-col")
          .action((x, c) => c.copy(labelCol = x))
          .text("Name of the continuous label column (e.g., 'score')"),
        opt[String]("train")
          .action((x, c) => c.copy(trainPath = x))
          .text("Path to training data (Parquet format)"),
        opt[String]("test")
          .action((x, c) => c.copy(testPath = x))
          .text("Path to test data (Parquet format)"),
        opt[String]("results")
          .action((x, c) => c.copy(resultsDir = x))
          .text("Results directory"),
        opt[Int]("max-iter")
          .action((x, c) => c.copy(maxIter = x))
          .text("Maximum iterations"),
        opt[Int]("max-depth")
          .action((x, c) => c.copy(maxDepth = x))
          .text("Maximum tree depth"),
        opt[Int]("num-trees")
          .action((x, c) => c.copy(numTrees = x))
          .text("Number of trees (for RF/GBT)"),
        opt[Double]("lr")
          .action((x, c) => c.copy(learningRate = x))
          .text("Learning rate"),
        opt[Double]("reg")
          .action((x, c) => c.copy(regParam = x))
          .text("Regularization parameter"),
        opt[Double]("elastic")
          .action((x, c) => c.copy(elasticNetParam = x))
          .text("ElasticNet parameter"),
        opt[Int]("seed")
          .action((x, c) => c.copy(seed = x))
          .text("Starting seed for consistency"),
        opt[Double]("eta")
          .action((x, c) => c.copy(eta = x))
          .text("XGBoost learning rate (eta)"),
        opt[Int]("num-round")
          .action((x, c) => c.copy(numRound = x))
          .text("XGBoost number of rounds"),
        opt[Int]("num-workers")
          .action((x, c) => c.copy(numWorkers = x))
          .text("XGBoost number of workers")
        // Eliminados: num-classes, objective
      )
    }
    OParser.parse(parser, args, Config()) match {
      case Some(config) =>
        run(config)
      case _ =>
        sys.exit(1)
    }
  }

  def run(config: Config): Unit = {
    val spark = SparkSession
      .builder()
      .appName("ReviewRegressor")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .getOrCreate()

    try {
      println(
        s"=== Starting ${config.model} for regression on label ${config.labelCol} ==="
      )
      println(s"Configuration: $config")

      // Load and prepare data
      val dataLoader = new DataLoader(spark)
      val (trainDf, testDf) =
        dataLoader.loadData(config.trainPath, config.testPath, config.labelCol)

      println(s"Training samples: ${trainDf.count()}")
      println(s"Test samples: ${testDf.count()}")

      // Get model
      val model = ModelFactory.getModel(config.model, config)

      // Train model
      println("\n=== Training model ===")
      val trainedModel = model.train(trainDf)

      // Evaluate
      println("\n=== Evaluating model ===")
      val evaluator = new Evaluator(spark) // Ya no necesita 'task'

      val trainMetrics = evaluator.evaluate(trainedModel, trainDf, "train")
      val testMetrics = evaluator.evaluate(trainedModel, testDf, "test")

      // Save results
      val resultSaver =
        new ResultSaver(config.resultsDir, config.model) // Ya no necesita 'task'
      resultSaver.saveMetrics(trainMetrics, testMetrics)
      resultSaver.saveModel(trainedModel, s"${config.model}_${config.labelCol}")

      println("\n=== Results Summary ===")
      println(f"Train RMSE: ${trainMetrics.rmse}%.4f")
      println(f"Test RMSE: ${testMetrics.rmse}%.4f")
      println(f"Test R-squared: ${testMetrics.r2}%.4f")

      println(
        s"\nResults saved to: ${config.resultsDir}/${config.model}/"
      )
    } finally {
      spark.stop()
    }
  }


}