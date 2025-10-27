
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