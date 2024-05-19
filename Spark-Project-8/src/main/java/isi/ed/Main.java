package isi.ed;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.callUDF;

public class Main {

    public static void main(String[] args) {
        Dataset<Row> df = DataHandler.LoadData();
        df = DataHandler.ProcessNumeric(df);
        Logistics lr = new Logistics(df);
        Logistics.PrintCoeffs(lr.lrModel);
        Dataset<Row> df_pred = Logistics.PrintPrediction(lr.lrModel, lr.df);
        df_pred.show();
        Logistics.AnalyzePredictions(df_pred.select("features","rawPrediction","probability","prediction"), lr.lrModel);
        df_pred = df_pred.drop("features", "rawPrediction")  // Remove unnecessary columns
                .withColumn("prob", callUDF("max_vector_element", df_pred.col("probability")))
                .drop("probability");
        df_pred.show();
        DataHandler.SaveData(df_pred);
    }
}