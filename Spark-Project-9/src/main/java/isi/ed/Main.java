package isi.ed;

import org.apache.commons.lang3.Range;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import javax.validation.constraints.NotNull;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static isi.ed.DataHandler.LoadData;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.length;

public class Main {

    public static void main(String[] args) {
        Dataset<Row> twoBooksAll1000_10Stem = LoadData("two-books-all-1000-10-stem");

        Dataset<Row> authors = twoBooksAll1000_10Stem.select("author").distinct();
        Dataset<Row> works = twoBooksAll1000_10Stem.select("work").distinct();

        //authors.show();
        //works.show();

        Dataset<Row> numAuthors = twoBooksAll1000_10Stem.groupBy("author").count();
        Dataset<Row> numWorks = twoBooksAll1000_10Stem.groupBy("work").count();

        //numAuthors.show();
        //numWorks.show();

        twoBooksAll1000_10Stem = twoBooksAll1000_10Stem.withColumn("length", length(col("content_stemmed")));
        //twoBooksAll1000_10Stem.show();

        Dataset<Row> avgLength = twoBooksAll1000_10Stem.groupBy("work").agg(
                org.apache.spark.sql.functions.avg("length").as("avg_length")
        );

        //avgLength.show();

        String sep = "[\\s\\p{Punct}—…”„]+";
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern(sep);
        var twoBooksAll1000_10Stem_tokenized = tokenizer.transform(twoBooksAll1000_10Stem);
        //twoBooksAll1000_10Stem_tokenized.show();

        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(2);           // Set the minimum number of documents in which a term must appear

        CountVectorizerModel countVectorizerModel = countVectorizer.fit(twoBooksAll1000_10Stem_tokenized);

        Dataset<Row> twoBooksAll1000_10Stem_BoW = countVectorizerModel.transform(twoBooksAll1000_10Stem_tokenized);
        twoBooksAll1000_10Stem_BoW.select("words", "features").show(5);
        twoBooksAll1000_10Stem_BoW.show();

        Row row = twoBooksAll1000_10Stem_BoW.first();
        SparseVector row6 = row.getAs("features");
        for (int index : row6.indices()) {
            // Ensure index is within the range of the vocabulary array
            if (index >= 0 && index < Arrays.stream(countVectorizerModel.vocabulary()).count() && index < row6.indices().length) {
                try {
                    System.out.println(countVectorizerModel.vocabulary()[index] + " -> " + row6.values()[index]);
                } catch (Exception e) {
                    System.out.println("Error: " + e);
                    break;
                }
            }
        }
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");
        StringIndexerModel labelModel = labelIndexer.fit(twoBooksAll1000_10Stem_BoW);
        twoBooksAll1000_10Stem_BoW = labelModel.transform(twoBooksAll1000_10Stem_BoW);
        twoBooksAll1000_10Stem_BoW.show();

        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini")  // lub entropy
                .setMaxDepth(30);

        DecisionTreeClassificationModel model = dt.fit(twoBooksAll1000_10Stem_BoW);

        Dataset<Row> twoBooksAll1000_10Stem_predictions = model.transform(twoBooksAll1000_10Stem_BoW);
        twoBooksAll1000_10Stem_predictions.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(twoBooksAll1000_10Stem_predictions);
        System.out.println("Test set accuracy: " + accuracy);

        evaluator = evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(twoBooksAll1000_10Stem_predictions);
        System.out.println("Test set f1: " + f1);

        SparseVector fi = (SparseVector) model.featureImportances();
        System.out.println(fi);
        System.out.println(Arrays.toString(fi.indices()));
        for (int index : fi.indices()) {
            // Ensure index is within the range of the vocabulary array
            if (index >= 0 && index < Arrays.stream(countVectorizerModel.vocabulary()).count() && index < fi.indices().length) {
                try {
                    System.out.println(countVectorizerModel.vocabulary()[index] + " -> " + fi.values()[index]);
                } catch (Exception e) {
                    System.out.println("Error: " + e);
                    break;
                }
            }
        }

        String[] filenames ={
                "two-books-all-1000-1-stem",
                "two-books-all-1000-3-stem",
                "two-books-all-1000-5-stem",
                "two-books-all-1000-10-stem",
                "five-books-all-1000-1-stem",
                "five-books-all-1000-3-stem",
                "five-books-all-1000-5-stem",
                "five-books-all-1000-10-stem",
        };

        //performGridSearchCV(DataHandler.spark, "two-books-all-1000-10-stem");
        //Dataset<Row> df_CV = performCV(DataHandler.spark, filenames);
        //df_CV.show();
        //NaiveBayesDemo naiveBayesDemo = new NaiveBayesDemo();
        //performGridSearchCVNaiveBayes(DataHandler.spark, "two-books-all-1000-1-stem");
        Dataset<Row> df_CV = performCVNaiveBayes(DataHandler.spark, filenames);
        df_CV.show();
    }
    private static void performGridSearchCV(SparkSession spark, String filename){
        var df = LoadData(filename);
        var splits = df.randomSplit(new double[]{0.8,0.2});
        var df_train = splits[0];
        var df_test = splits[1];

        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern("[\\s\\p{Punct}—…”„]+");

        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(2);

        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini")
                .setMaxDepth(30);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, countVectorizer, labelIndexer, decisionTreeClassifier});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(countVectorizer.vocabSize(), new int[] {100, 1000,10_000})
                .addGrid(decisionTreeClassifier.maxDepth(), new int[] {10, 20,30})
                .build();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3)  // Use 3+ in practice
                .setParallelism(8);

        CrossValidatorModel cvModel = cv.fit(df_train);
        PipelineModel bestModel = (PipelineModel) cvModel.bestModel();
        for(var s:bestModel.stages()){
            System.out.println(s);
        }
        System.out.println(Arrays.toString(cvModel.avgMetrics()));

        Dataset<Row> predictions = bestModel.transform(df_test);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("Test set f1: " + f1);

        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test set accuracy: " + accuracy);

        evaluator.setMetricName("weightedPrecision");
        double precision = evaluator.evaluate(predictions);
        System.out.println("Test set precision: " + precision);

        evaluator.setMetricName("weightedRecall");
        double recall = evaluator.evaluate(predictions);
        System.out.println("Test set recall: " + recall);
    }

    public static Dataset<Row> performCV(SparkSession spark, String[] filenames){

        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "F1",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "accuracy",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "weightedPrecision",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "weightedRecall",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "filename",
                        DataTypes.StringType,
                        true),
        });

        Dataset<Row> df_return = spark.createDataFrame(new ArrayList<Row>(), schema);

        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern("[\\s\\p{Punct}—…”„]+");

        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(2);

        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini")
                .setMaxDepth(20);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, countVectorizer, labelIndexer, decisionTreeClassifier});

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(new ParamGridBuilder().build())
                .setNumFolds(3)  // Use 3+ in practice
                .setParallelism(8);

        for(String filename : filenames){
            var df = LoadData(filename);
            var splits = df.randomSplit(new double[]{0.8,0.2});
            var df_train = splits[0];
            var df_test = splits[1];

            CrossValidatorModel cvModel = cv.fit(df_train);
            PipelineModel bestModel = (PipelineModel) cvModel.bestModel();

            Dataset<Row> predictions = bestModel.transform(df_test);

            evaluator.setMetricName("f1");
            double f1 = evaluator.evaluate(predictions);

            evaluator.setMetricName("accuracy");
            double accuracy = evaluator.evaluate(predictions);

            evaluator.setMetricName("weightedPrecision");
            double precision = evaluator.evaluate(predictions);

            evaluator.setMetricName("weightedRecall");
            double recall = evaluator.evaluate(predictions);

            // Create a new row
            Row newRow = RowFactory.create(f1, accuracy, precision, recall, filename);

            // Create a DataFrame with the new row
            List<Row> newRowList = new ArrayList<>();
            newRowList.add(newRow);
            Dataset<Row> newRowDF = spark.createDataFrame(newRowList, schema);

            // Add the new row to the existing DataFrame using union
            df_return = df_return.union(newRowDF);
        }

        return df_return;
    }

    private static void performGridSearchCVNaiveBayes(SparkSession spark, String filename){
        var df = LoadData(filename);
        var splits = df.randomSplit(new double[]{0.8,0.2});
        var df_train = splits[0];
        var df_test = splits[1];

        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern("[\\s\\p{Punct}—…”„]+");

        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(2);

        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        NaiveBayes nb = new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setSmoothing(0.2);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, countVectorizer, labelIndexer, nb});

        var scalaIterable = scala.jdk.CollectionConverters.
                IterableHasAsScala(Arrays.asList("multinomial", "gaussian")).asScala();

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(countVectorizer.vocabSize(), new int[] {100, 1000,5_000,10_000})
                .addGrid(nb.modelType(),scalaIterable )
                .build();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3)  // Use 3+ in practice
                .setParallelism(8);

        CrossValidatorModel cvModel = cv.fit(df_train);
        PipelineModel bestModel = (PipelineModel) cvModel.bestModel();
        for(var s:bestModel.stages()){
            System.out.println(s);
        }
        System.out.println(Arrays.toString(cvModel.avgMetrics()));

        Dataset<Row> predictions = bestModel.transform(df_test);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("Test set f1: " + f1);

        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test set accuracy: " + accuracy);

        evaluator.setMetricName("weightedPrecision");
        double precision = evaluator.evaluate(predictions);
        System.out.println("Test set precision: " + precision);

        evaluator.setMetricName("weightedRecall");
        double recall = evaluator.evaluate(predictions);
        System.out.println("Test set recall: " + recall);
    }

    public static Dataset<Row> performCVNaiveBayes(SparkSession spark, String[] filenames){

        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "F1",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "accuracy",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "weightedPrecision",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "weightedRecall",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "filename",
                        DataTypes.StringType,
                        true),
        });

        Dataset<Row> df_return = spark.createDataFrame(new ArrayList<Row>(), schema);

        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern("[\\s\\p{Punct}—…”„]+");

        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(2);

        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        NaiveBayes nb = new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setSmoothing(0.2)
                .setModelType("multinomial");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, countVectorizer, labelIndexer, nb});

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(new ParamGridBuilder().build())
                .setNumFolds(3)  // Use 3+ in practice
                .setParallelism(8);

        for(String filename : filenames){
            var df = LoadData(filename);
            var splits = df.randomSplit(new double[]{0.8,0.2});
            var df_train = splits[0];
            var df_test = splits[1];

            CrossValidatorModel cvModel = cv.fit(df_train);
            PipelineModel bestModel = (PipelineModel) cvModel.bestModel();

            Dataset<Row> predictions = bestModel.transform(df_test);

            evaluator.setMetricName("f1");
            double f1 = evaluator.evaluate(predictions);

            evaluator.setMetricName("accuracy");
            double accuracy = evaluator.evaluate(predictions);

            evaluator.setMetricName("weightedPrecision");
            double precision = evaluator.evaluate(predictions);

            evaluator.setMetricName("weightedRecall");
            double recall = evaluator.evaluate(predictions);

            // Create a new row
            Row newRow = RowFactory.create(f1, accuracy, precision, recall, filename);

            // Create a DataFrame with the new row
            List<Row> newRowList = new ArrayList<>();
            newRowList.add(newRow);
            Dataset<Row> newRowDF = spark.createDataFrame(newRowList, schema);

            // Add the new row to the existing DataFrame using union
            df_return = df_return.union(newRowDF);
        }

        return df_return;
    }
}