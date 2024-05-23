package isi.ed;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class Main {

    public static void main(String[] args) {
        Dataset<Row> twoBooksAll1000_10Stem = DataHandler.LoadData("two-books-all-1000-10-stem");
    }

}