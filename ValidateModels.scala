import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.PipelineModel
import java.io.{File, PrintWriter}

object ValidateModels {

  // --- Embedded classes for standalone execution ---

  case class Metrics(
      accuracy: Double,
      precision: Double,
      recall: Double,
      f1: Double,
      confusionMatrix: Array[Array[Double]],
      predictions: DataFrame
  )

  class Evaluator(spark: SparkSession, task: String) {
    import spark.implicits._

    def evaluate(
        model: Transformer,
        data: DataFrame,
        split: String
    ): Metrics = {
      val predictions = model.transform(data)

      val accuracyEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val accuracy = accuracyEvaluator.evaluate(predictions)

      val precisionEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("weightedPrecision")
      val precision = precisionEvaluator.evaluate(predictions)

      val recallEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("weightedRecall")
      val recall = recallEvaluator.evaluate(predictions)

      val f1Evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("f1")
      val f1 = f1Evaluator.evaluate(predictions)

      val predictionAndLabels = predictions
        .select($"prediction", $"label")
        .rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))

      val metrics = new MulticlassMetrics(predictionAndLabels)
      val confusionMatrix = metrics.confusionMatrix.toArray
        .grouped(metrics.confusionMatrix.numCols)
        .toArray

      println(s"\n=== $split Metrics ===")
      println(f"Accuracy: $accuracy%.4f")
      println(f"Precision: $precision%.4f")
      println(f"Recall: $recall%.4f")
      println(f"F1 Score: $f1%.4f")
      println("\nConfusion Matrix:")
      confusionMatrix.foreach(row => println(row.mkString("\t")))

      Metrics(accuracy, precision, recall, f1, confusionMatrix, predictions)
    }
  }

  class DataLoader(spark: SparkSession) {
    import spark.implicits._

    def loadData(
        trainPath: String,
        testPath: String,
        task: String
    ): (DataFrame, DataFrame) = {
      val trainRaw = spark.read.parquet(trainPath)
      val testRaw = spark.read.parquet(testPath)

      val (trainProcessed, testProcessed) = task.toLowerCase match {
        case "category" =>
          prepareForCategory(trainRaw, testRaw)
        case "sentiment" =>
          prepareForSentiment(trainRaw, testRaw)
        case _ => throw new IllegalArgumentException(s"Unknown task: $task")
      }

      (trainProcessed, testProcessed)
    }

    private def prepareForCategory(
        trainDf: DataFrame,
        testDf: DataFrame
    ): (DataFrame, DataFrame) = {
      val featureCols = trainDf.columns.filter(c =>
        c.startsWith("title_emb_") || c.startsWith("review_emb_")
      ) ++ Array("sentiment_encoded")

      val assembler = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol("raw_features")

      val trainAssembled = assembler.transform(trainDf)
      val testAssembled = assembler.transform(testDf)

      // The scaler is fitted on the training data and then used to transform the test data.
      // This is a crucial preprocessing step, not model retraining.
      val scaler = new StandardScaler()
        .setInputCol("raw_features")
        .setOutputCol("features")
        .setWithStd(true)
        .setWithMean(true)
        .fit(trainAssembled)

      val trainScaled = scaler
        .transform(trainAssembled)
        .withColumn("label", col("category_encoded").cast("double"))
        .select("features", "label")

      val testScaled = scaler
        .transform(testAssembled)
        .withColumn("label", col("category_encoded").cast("double"))
        .select("features", "label")

      (trainScaled, testScaled)
    }

    private def prepareForSentiment(
        trainDf: DataFrame,
        testDf: DataFrame
    ): (DataFrame, DataFrame) = {
      val featureCols = trainDf.columns.filter(c =>
        c.startsWith("title_emb_") || c.startsWith("review_emb_")
      ) ++ Array("category_encoded")

      val assembler = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol("raw_features")

      val trainAssembled = assembler.transform(trainDf)
      val testAssembled = assembler.transform(testDf)

      // The scaler is fitted on the training data and then used to transform the test data.
      // This is a crucial preprocessing step, not model retraining.
      val scaler = new StandardScaler()
        .setInputCol("raw_features")
        .setOutputCol("features")
        .setWithStd(true)
        .setWithMean(true)
        .fit(trainAssembled)

      val trainScaled = scaler
        .transform(trainAssembled)
        .withColumn("label", col("sentiment_encoded").cast("double"))
        .select("features", "label")

      val testScaled = scaler
        .transform(testAssembled)
        .withColumn("label", col("sentiment_encoded").cast("double"))
        .select("features", "label")

      (trainScaled, testScaled)
    }
  }

  // --- Main logic ---

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      println(
        "Usage: scala ValidateModels.scala <path_to_train_data.parquet> <path_to_test_data.parquet>"
      )
      println("Please provide paths to both training and testing data files.")
      sys.exit(1)
    }

    val trainDataPath = args(0)
    val testDataPath = args(1)

    val spark = SparkSession.builder
      .appName("ValidateModels")
      .master("local[*]")
      .config("spark.driver.memory", "10g")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val resultsDir = "results"

    val modelsToTest =
      Seq("LogisticRegression", "RandomForest", "MLP", "NaiveBayes")
    val tasksToTest = Seq("category", "sentiment")

    val dataLoader = new DataLoader(spark)

    tasksToTest.foreach { task =>
      println(s"===== Validating task: $task =====")
      // We load the training data here only to fit the scaler, ensuring test data is processed correctly.
      val (_, testData) = dataLoader.loadData(trainDataPath, testDataPath, task)
      val evaluator = new Evaluator(spark, task)

      modelsToTest.foreach { modelName =>
        println(s"--- Validating model: $modelName ---")
        val modelPath = s"$resultsDir/${modelName}_$task/models/$modelName"

        if (new File(modelPath).exists()) {
          val model = PipelineModel.load(modelPath)

          val testMetrics = evaluator.evaluate(model, testData, "test")

          saveTestMetrics(resultsDir, modelName, task, testMetrics)
        } else {
          println(s"Model not found at $modelPath, skipping.")
        }
      }
    }

    spark.stop()
  }

  def saveTestMetrics(
      resultsDir: String,
      modelName: String,
      task: String,
      testMetrics: Metrics
  ): Unit = {
    val outputDir = s"$resultsDir/${modelName}_$task"
    new File(outputDir).mkdirs()

    val path = s"$outputDir/metrics_summary.csv"
    val writer = new PrintWriter(new File(path))
    try {
      writer.write("split,accuracy,precision,recall,f1\n")
      // Write dummy train metrics for compatibility with analyze_results.py
      writer.write(f"train,0.0,0.0,0.0,0.0\n")
      writer.write(
        f"test,${testMetrics.accuracy}%.6f,${testMetrics.precision}%.6f,${testMetrics.recall}%.6f,${testMetrics.f1}%.6f\n"
      )
      println(s"Metrics summary saved to: $path")
    } finally {
      writer.close()
    }
  }
}
