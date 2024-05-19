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
    static SparkSession spark = SparkSession.builder()
            .appName("LogisticRegressionOnExam")
            .master("local")
            .getOrCreate();

    public static Dataset<Row> LoadData(){
        spark.udf().register(  "max_vector_element",new Logistics.MaxVectorElement(),DataTypes.DoubleType);
        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "ImieNazwisko",
                        DataTypes.StringType,
                        true),
                DataTypes.createStructField(
                        "OcenaC",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "DataC",
                        DataTypes.DateType,
                        true),
                DataTypes.createStructField(
                        "OcenaCPP",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "Egzamin",
                        DataTypes.DoubleType,
                        true),
        });

        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .option("delimiter", ";")
                .schema(schema)
                .load("src/main/resources/egzamin-cpp.csv");

        System.out.println("Excerpt of the dataframe content:");
        df.show(20);

        return df;
    }

    public static void SaveData(Dataset<Row> df){
        df = df.repartition(1);
        df.write()
                .format("csv")
                .option("header", true)
                .option("delimiter",",")
                .mode(SaveMode.Overwrite)
                .save("./src/main/resources/egzamin-cpp");
    }

    public static @NotNull Dataset<Row> ProcessNumeric(Dataset<Row> df){
        df = df.withColumn("Timestamp", unix_timestamp(col("DataC"), "yyyy-MM-dd"));
        // create a columns egzamin2 where it will hold a result of 1 if exam result is >= 3.0 and 0 otherwise, using SQL IF function
        df = df.withColumn("Wynik", expr("IF(Egzamin>=3.0, 1, 0)"));

        df.show(20);
        return df;
    }
}
