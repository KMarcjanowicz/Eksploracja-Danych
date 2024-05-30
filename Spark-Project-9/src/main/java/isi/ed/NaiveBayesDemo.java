package isi.ed;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static isi.ed.DataHandler.spark;

public class NaiveBayesDemo {
    NaiveBayesDemo(){
        System.out.println("NaiveBayesDemo");
        StructType schema = new StructType()
                .add("author", DataTypes.StringType, false)
                .add("content", DataTypes.StringType, false);
        List<Row> rows = Arrays.asList(
                RowFactory.create("Ala","aaa aaa bbb ccc"),
                RowFactory.create("Ala","aaa bbb ddd"),
                RowFactory.create("Ala","aaa bbb"),
                RowFactory.create("Ala","aaa bbb bbb"),
                RowFactory.create("Ola","aaa ccc ddd"),
                RowFactory.create("Ola","bbb ccc ddd"),
                RowFactory.create("Ola","ccc ddd eee")
        );

        var df = spark.createDataFrame(rows,schema);

        String sep = "[\\s\\p{Punct}—…”„]+";
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern(sep);
        df = tokenizer.transform(df);
        df.show();

        System.out.println("-----------");
        // Convert to BoW with CountVectorizer
        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(1)     // Set the minimum number of documents in which a term must appear
                ;

        // Fit the model and transform the DataFrame
        CountVectorizerModel countVectorizerModel = countVectorizer.fit(df);
        df = countVectorizerModel.transform(df);


        // Prepare the data: index the label column
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        StringIndexerModel labelModel = labelIndexer.fit(df);
        df = labelModel.transform(df);
        df.show();

        NaiveBayes nb = new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setModelType("multinomial")
                .setSmoothing(0.00)
                ;
        System.out.println(nb.explainParams());

        NaiveBayesModel model = nb.fit(df);

        String[] vocab = countVectorizerModel.vocabulary();
        String[] labels = labelModel.labels();

        for(int i = 0; i < labels.length; i++){
            for(int j = 0; j < vocab.length; j++){
                //System.out.println(countVectorizerModel.vocabulary()[index] + " -> " + fi.values()[index]);
                System.out.printf("P(%s|%s)=%.6f (log=%.6f)%n", vocab[j], labels[i], model.theta().apply(i, j), Math.log(model.theta().apply(i, j)));
            }
        }
        for(int i = 0; i < labels.length; i++){
            for(int j = 0; j < vocab.length; j++){
                //System.out.println(countVectorizerModel.vocabulary()[index] + " -> " + fi.values()[index]);
                System.out.printf("P(%s)=%.6f (log=%.6f)%n", labels[i], model.pi().apply(i), Math.log(model.pi().apply(i)));
            }
        }

        // bbb ddd ccc ddd
        var testData = new DenseVector(new double[]{1,0,2,1,0});
        var proba = model.predictRaw(testData);
        System.out.println("Pr:["+ Math.exp(proba.apply(0))+", "+Math.exp(proba.apply(1)));
        var predLabel = model.predict(testData);
        System.out.println(predLabel);

        // Wektor prawdopodobieństw dla klasy 0
        double[] p0 = {0.2, 0.3, 0.1, 0.4};
        // Wektor prawdopodobieństw dla klasy 1
        double[] p1 = {0.3, 0.2, 0.4, 0.1};
        // Wektor danych
        double[] x = {1, 0, 2, 1};

        // Logarytmy prawdopodobieństw dla klasy 0
        double[] logP0 = new double[p0.length];
        for (int i = 0; i < p0.length; i++) {
            logP0[i] = Math.log(p0[i]);
        }

        // Logarytmy prawdopodobieństw dla klasy 1
        double[] logP1 = new double[p1.length];
        for (int i = 0; i < p1.length; i++) {
            logP1[i] = Math.log(p1[i]);
        }

        // Obliczenie log(p0(x))
        double logP0x = 0.0;
        for (int i = 0; i < x.length; i++) {
            logP0x += x[i] * logP0[i];
        }

        // Obliczenie log(p1(x))
        double logP1x = 0.0;
        for (int i = 0; i < x.length; i++) {
            logP1x += x[i] * logP1[i];
        }

        // Wynik logarytmiczny i prawdopodobieństwo po eksponentacji
        System.out.printf(Locale.US,"log(p0)=%g p0=%g log(p1)=%g p1=%g%n",
                logP0x, Math.exp(logP0x),
                logP1x, Math.exp(logP1x));

        // Wynik klasyfikacji
        System.out.println("Wynik klasyfikacji:" + (logP0x > logP1x ? 0 : 1));
    }
}
