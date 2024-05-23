package isi.ed;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.jetbrains.annotations.NotNull;

import static org.apache.spark.sql.functions.*;

public class DataHandler {
    public static SparkSession spark = SparkSession.builder()
            .appName("LogisticRegressionOnExam")
            .master("local")
            .getOrCreate();

    public static Dataset<Row> LoadData(String filename){
        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "author",
                        DataTypes.StringType,
                        true),
                DataTypes.createStructField(
                        "work",
                        DataTypes.StringType,
                        true),
                DataTypes.createStructField(
                        "content",
                        DataTypes.StringType,
                        true),
                DataTypes.createStructField(
                        "content_stemmed",
                        DataTypes.StringType,
                        true),
        });

        Dataset<Row> df = spark.read().format("csv")
                .option("header", true)
                .option("delimiter",",")
                .option("quote","\'")
                .option("inferschema","true")
                .load("src/main/resources/" + filename + ".csv");

        System.out.println("Excerpt of the dataframe content:");
        df.show(20);

        return df;
    }

    public static void SaveData(Dataset<Row> df, int partitions, String filename){
        df = df.repartition(partitions);
        df.write()
                .format("csv")
                .option("header", true)
                .option("delimiter",",")
                .option("quote","\'")
                .mode(SaveMode.Overwrite)
                .save("./src/main/resources/" + filename + ".csv");
    }

    public static @NotNull Dataset<Row> ProcessNumeric(Dataset<Row> df){
        df = df.withColumn("Timestamp", unix_timestamp(col("DataC"), "yyyy-MM-dd"));
        // create a columns egzamin2 where it will hold a result of 1 if exam result is >= 3.0 and 0 otherwise, using SQL IF function
        df = df.withColumn("Wynik", expr("IF(Egzamin>=3.0, 1, 0)"));

        df.show(20);
        return df;
    }
}
