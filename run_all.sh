#!/bin/bash
MODELS=("RandomForest" "LogisticRegression" "GradientBoosting" "NaiveBayes" "MLP")
TASKS=("category" "sentiment")

JAR_FILE="target/scala-2.12/ReviewClassifier-0.1.0.jar"

if [ ! -f "$JAR_FILE" ]; then
    echo "nof ile found at $JAR_FILE"
    exit 1
fi

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "  Starting Run: Model = $model, Task = $task"
    echo "======================================================================"
    echo ""

    spark-submit \
      --class com.reviewclassifier.Main \
      --master 'local[*]' \
      --driver-memory 22g \
      --files src/main/resources/log4j2.properties \
    --driver-java-options "-Dlog4j.configurationFile=log4j2.properties" \
      "$JAR_FILE" \
      --model "$model" \
      --task "$task"

    if [ $? -ne 0 ]; then
      echo ""
      echo "----------------------------------------------------------------------"
      echo "  ERROR: Run failed for Model = $model, Task = $task"
      echo "----------------------------------------------------------------------"
      echo ""
    else
      echo ""
      echo "----------------------------------------------------------------------"
      echo "  SUCCESS: Run completed for Model = $model, Task = $task"
      echo "----------------------------------------------------------------------"
      echo ""
    fi
  done
done

echo "All runs completed."
