package isi.ed;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;

import static isi.ed.Plotting.plot_scatter;
import static org.apache.spark.sql.functions.*;

public class JoinUserRatings {
    public static void JoinUserRatings(Dataset<Row> df_users, Dataset<Row> df_ratings){

        System.out.println("Users dataframe: ");
        df_users.show(5);

        System.out.println("Ratings dataframe: ");
        df_ratings.show(5);

        Dataset<Row> df_ur = df_users.join(df_ratings,"userId","inner");
        df_ur.show(5);

        Dataset<Row> dr_ur_grouped = df_ur.groupBy("email").agg(
                avg("rating").alias("avg_rating"),
                count("email").alias("count")
        ).orderBy(col("avg_rating").desc());

        dr_ur_grouped.show(5);

        var x = dr_ur_grouped.select("avg_rating").as(Encoders.DOUBLE()).collectAsList();
        var y = dr_ur_grouped.select("count").as(Encoders.DOUBLE()).collectAsList();

        plot_scatter(x, y, "Number of ratings vs. average rating");
    }
}
