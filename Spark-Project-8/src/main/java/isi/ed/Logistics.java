package isi.ed;


import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.api.java.UDF1;

import java.text.DecimalFormat;
import java.util.Arrays;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

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
}