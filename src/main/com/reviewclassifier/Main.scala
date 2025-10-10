package com.reviewclassifier

import org.apache.spark.sql.SparkSession
import scopt.OParser
import com.reviewclassifier.models._
import com.reviewclassifier.utils._

case class Config(
    model: String = "RandomForest",
    task: String = "category", // category or sentiment
    trainPath: String = "data/train.csv",
    testPath: String = "data/test.csv",
    resultsDir: String = "results",
    maxIter: Int = 100,
    maxDepth: Int = 10,
    numTrees: Int = 100,
    learningRate: Double = 0.1,
    regParam: Double = 0.01,
    elasticNetParam: Double = 0.0
)

object Main {
  def main(args: Array[String]): Unit = {
    val builder = OParser.builder[Config]
    val parser = {
      import builder._
      OParser.sequence(
        programName("ReviewClassifier"),
        head("Review Classifier", "0.1.0"),
        opt[String]('m', "model")
          .action((x, c) => c.copy(model = x))
          .text(
            "Model: RandomForest, LogisticRegression, GradientBoosting, NaiveBayes, MLP"
          ),
        opt[String]('t', "task")
          .action((x, c) => c.copy(task = x))
          .text("Task: category or sentiment"),
        opt[String]("train")
          .action((x, c) => c.copy(trainPath = x))
          .text("Path to training CSV"),
        opt[String]("test")
          .action((x, c) => c.copy(testPath = x))
          .text("Path to test CSV"),
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
          .text("ElasticNet parameter")
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
      .appName("ReviewClassifier")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .getOrCreate()

    try {
      println(
        s"=== Starting ${config.model} for ${config.task} classification ==="
      )
      println(s"Configuration: $config")

      // Load and prepare data
      val dataLoader = new DataLoader(spark)
      val (trainDf, testDf) =
        dataLoader.loadData(config.trainPath, config.testPath, config.task)

      println(s"Training samples: ${trainDf.count()}")
      println(s"Test samples: ${testDf.count()}")

      // Get model
      val model = ModelFactory.getModel(config.model, config)

      // Train model
      println("\n=== Training model ===")
      val trainedModel = model.train(trainDf)

      // Evaluate
      println("\n=== Evaluating model ===")
      val evaluator = new Evaluator(spark, config.task)

      val trainMetrics = evaluator.evaluate(trainedModel, trainDf, "train")
      val testMetrics = evaluator.evaluate(trainedModel, testDf, "test")

      // Save results
      val resultSaver =
        new ResultSaver(config.resultsDir, config.model, config.task)
      resultSaver.saveMetrics(trainMetrics, testMetrics)
      resultSaver.saveModel(trainedModel, s"${config.model}_${config.task}")

      println("\n=== Results Summary ===")
      println(s"Train Accuracy: ${trainMetrics.accuracy}")
      println(s"Test Accuracy: ${testMetrics.accuracy}")
      println(s"Test F1 Score: ${testMetrics.f1}")

      println(
        s"\nResults saved to: ${config.resultsDir}/${config.model}_${config.task}/"
      )
    } finally {
      spark.stop()
    }
  }
}
