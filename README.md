# Scala NLP and Big Data Project: Review Classifier

This project implements and evaluates several machine learning models using Apache Spark to classify user reviews based on sentiment or category.

## Requirements

- Java 17: The project is built using Java 17. You can manage Java versions using a tool like [SDKMAN!](https://sdkman.io/).
- sbt: The project uses sbt as the build tool. It will be installed automatically by most IDEs, or you can install it manually.
- Spark 3.5.x: The models run on Apache Spark. The easiest way to install Spark is also via [SDKMAN!](https://sdkman.io/) (`sdk install spark`).

## Setup and Data Preparation

1.  Clone the Repository
    ```bash
    git clone <repository-url>
    cd Scala-NLP-Big-Data
    ```

2.  Add Raw Data
    Place your raw data file, `embeddings_separados_RESTMEX.csv`, in the root directory of the project.

3.  Prepare the Data
    The following steps will split your raw data into training and testing sets and then convert them to the efficient Parquet format.

    a. Split into Train/Test Sets (80/20)
       This command splits the raw CSV into `data/train.csv` and `data/test.csv`.
       ```bash
       head -n 1 embeddings_separados_RESTMEX.csv > header.csv && \
       tail -n +2 embeddings_separados_RESTMEX.csv > data.csv && \
       shuf data.csv > shuffled_data.csv && \
       head -n 229935 shuffled_data.csv > train_data.csv && \
       tail -n 57484 shuffled_data.csv > test_data.csv && \
       cat header.csv train_data.csv > data/train.csv && \
       cat header.csv test_data.csv > data/test.csv && \
       rm header.csv data.csv shuffled_data.csv train_data.csv test_data.csv
       ```

    b. **Convert to Parquet**
       This command runs a one-time conversion script to create `data/train.parquet` and `data/test.parquet`.
       ```bash
       sbt "runMain com.reviewclassifier.CsvToParquet"
       ```

## How to Run

1.  Build the Project
    First, package the application into a "fat" JAR that can be submitted to Spark.
    ```bash
    sbt assembly
    ```

2.  Run a Single Model
    Use the `spark-submit` command to run a specific model and task. Remember to quote the `--master` argument.
    ```bash
    spark-submit \
      --class com.reviewclassifier.Main \
      --master 'local[*]' \
      --driver-memory 8g \
      target/scala-2.12/ReviewClassifier-0.1.0.jar \
      --model RandomForest \
      --task category
    ```

3.  Run All Models
    A helper script is provided to run all model and task combinations sequentially.

    a. Make the script executable:
       ```bash
       chmod +x run_all.sh
       ```

    b. **Execute the script:**
       ```bash
       ./run_all.sh
       ```

## Adding or Modifying Models

The project is designed to be extensible. Follow these steps to add a new model.

### Step 1: Create a Model Class

Create a new Scala file in `src/main/scala/com/reviewclassifier/models/`. Your new class must extend the `BaseModel` trait.

Here is a template to get you started:

```scala
package com.reviewclassifier.models

import com.reviewclassifier.Config
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.classification.DecisionTreeClassifier // Example import
import org.apache.spark.sql.DataFrame

class MyNewModel(config: Config) extends BaseModel {
  override def train(data: DataFrame): Transformer = {
    // 1. Define your Spark ML model and set its parameters
    val myModel = new DecisionTreeClassifier() // Example
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(config.maxDepth) // Use parameters from the Config

    // 2. Create a Pipeline
    val pipeline = new Pipeline().setStages(Array(myModel))

    // 3. Fit the pipeline and return the trained model
    pipeline.fit(data)
  }

  override def getName: String = "MyNewModel"
}
```

### Step 2: Register the Model in the Factory

Open the file `src/main/scala/com/reviewclassifier/models/BaseModel.scala` and add your new model to the `ModelFactory`.

```scala
// In ModelFactory.getModel ...
object ModelFactory {
  def getModel(modelName: String, config: Config): BaseModel = {
    modelName.toLowerCase match {
        // ... existing models
        case "mynewmodel" => new MyNewModel(config) // Add your new model here
        case _ => throw new IllegalArgumentException(s"Unknown model: $modelName")
    }
  }
}
```

### Step 3: Add Hyperparameters (Optional)

If your new model requires hyperparameters that are not already in the `Config` class, open `src/main/scala/com/reviewclassifier/Main.scala` and:

1.  Add the new parameter to the `Config` case class.
2.  Add a corresponding `opt[...]` to the `OParser` so you can set it from the command line.

### Step 4: Recompile and Run
