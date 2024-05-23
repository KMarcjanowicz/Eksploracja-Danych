package isi.ed;

import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import scala.Tuple2;

import java.io.IOException;

import static isi.ed.DataHandler.spark;
import static isi.ed.Logistics.*;
import static org.apache.spark.sql.functions.callUDF;

public class Main {

    public static void main(String[] args) throws PythonExecutionException, IOException {
        Dataset<Row> df = DataHandler.LoadData();
        df = DataHandler.ProcessNumeric(df);
        Logistics lr = new Logistics(df);
        Logistics.PrintCoeffs(lr.lrModel);
        Dataset<Row> df_pred = Logistics.PrintPrediction(lr.lrModel, lr.df);
        df_pred.show();
        Logistics.AnalyzePredictions(df_pred.select("features","rawPrediction","probability","prediction"), lr.lrModel);
        Dataset<Row> df_trans = df_pred.drop("prediction", "rawPrediction", "probability");
        df_pred = df_pred.drop("features", "rawPrediction")  // Remove unnecessary columns
                .withColumn("prob", callUDF("max_vector_element", df_pred.col("probability")))
                .drop("probability");
        df_pred.show();
        DataHandler.SaveData(df_pred, 5);
        Tuple2<LogisticRegressionModel, Dataset<Row>> t2 = TrainAndTest(df_trans);
        LogisticRegressionModel model = t2._1();
        Dataset<Row> test = t2._2();
        BinaryLogisticRegressionTrainingSummary trainingSummary = model.binarySummary();
        double[] objectiveHistory = trainingSummary.objectiveHistory();
        plotObjectiveHistory(objectiveHistory);
        Dataset<Row> roc = trainingSummary.roc();
        roc.show();
        plotROC(roc);
        showMeasures(trainingSummary);

        Dataset<Row> df_fmeasures = trainingSummary.fMeasureByThreshold();
        df_fmeasures.offset(35).show();

        double maxFMeasure = df_fmeasures.select(functions.max("F-Measure")).head().getDouble(0);
        Row bestThresholdRow = df_fmeasures.where(df_fmeasures.col("F-Measure").equalTo(maxFMeasure)).select("threshold").head();
        double bestThreshold = bestThresholdRow.getDouble(0);
        System.out.println("Best threshold: " + bestThreshold);
        model.setThreshold(bestThreshold);

        df_fmeasures.show();

        evaluateModel(test, model);

        addClassificationToGrid(spark, model);
    }

}