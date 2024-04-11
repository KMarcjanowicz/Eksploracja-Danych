package isi.ed;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import java.util.List;

import static org.apache.spark.sql.functions.*;

public class JoinUserTags {
    public static void JoinUserTags(Dataset<Row> df_users, Dataset<Row> df_tags){
        //Spark session for the View logic
        SparkSession spark = SparkSession.builder()
                .appName("JoinMoviesGenres")
                .master("local")
                .getOrCreate();

        df_users.createOrReplaceTempView("users"); //vs. GlobalTempView
        df_tags.createOrReplaceTempView("tags");
        df_users.show();
        df_tags.show();

        String query = """
            SELECT * FROM users u INNER JOIN tags t ON  t.userId = u.userId;
        """;

        Dataset<Row> df_ut = spark.sql(query);
        df_ut = df_ut.drop("userId");

        Dataset<Row> df_ut_count = df_ut.groupBy("email").count();
        df_ut_count.show();

        Dataset<Row> df_tags_joined = df_ut.groupBy("email").agg(
                functions.concat_ws(" ", functions.collect_list("tag")).alias("tags")
        );
        System.out.println("Joined tags, grouped by the email: ");
        df_tags_joined.show();

        List<Row> list = df_tags_joined.select("tags").collectAsList();
        System.out.println("Printing first 5 results from the list of tags");
        for(int i = 0; i < 5; i++){
            System.out.println(list.get(i));
        }
    }
}
