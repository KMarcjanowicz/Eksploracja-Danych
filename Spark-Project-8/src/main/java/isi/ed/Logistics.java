package isi.ed;


import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import scala.Tuple2;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static isi.ed.DataHandler.LoadGrid;
import static org.apache.spark.sql.functions.*;

public class Logistics {

    LogisticRegression lr;
    LogisticRegressionModel lrModel;
    Dataset<Row> df;
    Logistics(Dataset<Row> _df)
    {
        lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.1)
                .setElasticNetParam(0)
                .setFeaturesCol("features")
                .setLabelCol("Wynik");

        String[] labels = {"OcenaC", "Timestamp", "OcenaCPP"};

        VectorAssembler va = new VectorAssembler()
                .setInputCols(labels)
                .setOutputCol("features");

        df = va.transform(_df);

        lrModel = lr.fit(df);
    }

    public static void PrintCoeffs(LogisticRegressionModel lrModel){
        DecimalFormat df = new DecimalFormat("#.######"); // Define the format
        //logit(zdal) = 0.719097*OcenaC + -0.000000*timestamp + 0.993461*OcenaCPP + 118.340611
        // get coefficient at index 0 from lr.coefficients()
        String[] labels = {"OcenaC", "Timestamp", "OcenaCPP"};
        var c = lrModel.coefficients().toArray();
        System.out.println("logit(zdal) = " + Double.parseDouble(df.format(c[0])) + "*OcenaC + " + Double.parseDouble(df.format(c[1])) + "*Timestamp + " + Double.parseDouble(df.format(c[2])) + "*OcenaCPP + " + lrModel.intercept());

        for (int i = 0; i < c.length; i++) {
            System.out.println("Coefficient at index " + i + " is " + c[i]);

            //Wzrost OcenaC o 1 zwiększa logit o 0.719097, a szanse zdania razy 2.052578 czyli o 105.257821%
            //Wzrost DataC o 1 dzień zwiększa logit o -0.000000,a  szanse zdania razy 0.992648 czyli o -0.735167%
            //Wzrost OcenaCPP o 1 zwiększa logit o 0.719097,a szanse zdania razy 2.700564 czyli o 170.056381%

            System.out.println("Wzrost " + labels[i] + " o 1 zwiększa logit o " + Double.parseDouble(df.format(c[i])) + ", a szanse zdania razy " + Double.parseDouble(df.format(Math.exp(c[i]))) + " czyli o " + Double.parseDouble(df.format((Math.exp(c[i])-1)*100)) + "%");
        }
    }
    public static Dataset<Row> PrintPrediction(LogisticRegressionModel lrModel, Dataset<Row> df_trans){
        Dataset<Row> df_with_predictions=lrModel.transform(df_trans);
        df_with_predictions.select("features","rawPrediction","probability","prediction").show();
        return df_with_predictions;

    }

    public static void AnalyzePredictions(Dataset<Row> dfPredictions, LogisticRegressionModel lrModel) {
        // Register UDF to calculate logit
        dfPredictions.foreach((ForeachFunction<Row>) row -> {

            Vector rawPrediction = row.getAs("rawPrediction");

            Vector features = row.getAs("features");
            double logit = features.dot(lrModel.coefficients()) + lrModel.intercept();


            // Calculate probabilities
            double probability1 = 1 / (1 + Math.exp(-logit));
            double probability0 = 1 - probability1;

            // Print results
            System.out.println("Raw Prediction: " + rawPrediction);
            System.out.println("Calculated Probability P(0): " + probability0);
            System.out.println("Calculated Probability P(1): " + probability1);
            System.out.println("Selected Probability: " + Math.max(probability0, probability1));
        });
    }

    //write me a function trainandtest which does:
    // takes a datase<ROW> and transforms it using vector assempleer
    // splits it into two sets: train adn test
    // trains a logistic regression model on the train set
    // tests the model on the test set
    // returns the model
    static Tuple2<LogisticRegressionModel, Dataset<Row>> TrainAndTest(Dataset<Row> df){
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.1)
                .setElasticNetParam(0)
                .setFeaturesCol("features")
                .setLabelCol("Wynik");

        String[] labels = {"OcenaC", "Timestamp", "OcenaCPP"};

        VectorAssembler va = new VectorAssembler()
                .setInputCols(labels)
                .setOutputCol("features");

        Dataset<Row>[] splits = df.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        return new Tuple2<>(lr.fit(train), test);
    }

    public static Dataset<Row> AddPropability(Dataset<Row> df){
        df = df.withColumn("prob",callUDF("max_vector_element",col("probability")));
        df.show();
        return df;
    }

    static class MaxVectorElement implements UDF1<Vector,Double> {
        @Override
        public Double call(Vector vector) throws Exception {
            return vector.toArray()[vector.argmax()];
        }
    }

    public static void plotObjectiveHistory(double[] objectiveHistory){
        List<Integer> x = IntStream.range(0, objectiveHistory.length).boxed().collect(Collectors.toList());
        List<Double> y = Arrays.stream(objectiveHistory).boxed().collect(Collectors.toList());
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.plot().add(x, y).label("Objective history");
        plt.xlabel("Iteration");
        plt.ylabel("Objective");
        plt.title("Objective history");
        plt.legend();
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    static void plotROC(Dataset<Row> roc){
        List<Double> tpr = roc.select("TPR").as(Encoders.DOUBLE()).collectAsList();
        List<Double> fpr = roc.select("FPR").as(Encoders.DOUBLE()).collectAsList();Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.plot().add(fpr, tpr).label("ROC curve");
        plt.xlabel("False Positive Rate");
        plt.ylabel("True Positive Rate");
        plt.title("ROC curve");
        plt.legend();
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    public static void showMeasures(BinaryLogisticRegressionTrainingSummary summary){
        System.out.println("Accuracy: " + summary.accuracy());
        System.out.println("FPR: " + summary.weightedFalsePositiveRate());
        System.out.println("TPR: " + summary.weightedTruePositiveRate());
        System.out.println("Precision: " + summary.weightedPrecision());
        System.out.println("Recall: " + summary.weightedRecall());
        System.out.println("F-measure: " + summary.weightedFMeasure());
    }
    public static void evaluateModel(Dataset<Row> df, LogisticRegressionModel model){

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("Wynik")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(model.transform(df));
        System.out.println("Accuracy: " + accuracy);

        evaluator.setMetricName("weightedPrecision");
        double weightedPrecision = evaluator.evaluate(model.transform(df));
        System.out.println("Weighted Precision: " + weightedPrecision);

        evaluator.setMetricName("weightedRecall");
        double weightedRecall = evaluator.evaluate(model.transform(df));
        System.out.println("Weighted Recall: " + weightedRecall);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(model.transform(df));
        System.out.println("F1: " + f1);
    }

//    Napisz funkcje
//
//    void addClassificationToGrid(SparkSession spark, LogisticRegressionModel lrModel)
//    która:
//
//    Wczyta zbiór danych grid.csv
//    Przetworzy daty, tak aby stały się wartościami numerycznymi
//    Skonfiguruje VectorAssembler
//    Wywoła funkcję predykcji zmiennej lrModel
//    Usunie nadmiarowe kolumny
//    Za pomocą funkcji IF() SQL lub zarejestrowanej funkcji użytkownika UDF dokona konwersji etykiet 0→Nie zdał oraz 1→Zdał
//    Wyświetli wynik
//    Zapisze w pliku grid-with-classification.csv

    static void addClassificationToGrid(SparkSession spark, LogisticRegressionModel lrModel){

        Dataset<Row> df = LoadGrid();
        df = df.withColumn("Timestamp", unix_timestamp(col("DataC"), "yyyy-MM-dd"));

        df.show();

        String[] labels = {"OcenaC", "Timestamp", "OcenaCPP"};

        VectorAssembler va = new VectorAssembler()
                .setInputCols(labels)
                .setOutputCol("features");

        df = va.transform(df);
        Dataset<Row> predictions = lrModel.transform(df);

        predictions = predictions.withColumn("Wynik", expr("IF(prediction=1.0, 'Zdal', 'Nie zdal')"));
        predictions.drop("prediction", "rawPrediction", "probability", "Timestamp", "features").show();

        predictions.write()
                .format("csv")
                .option("header", true)
                .option("delimiter",",")
                .mode(SaveMode.Overwrite)
                .save("./src/main/resources/grid-with-classification-cpp");
    }
}