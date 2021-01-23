import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get('DEFAULT','AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('DEFAULT','AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    Description: This function is used for creating a SparkSession

    Returns:
        SparkSession: The Entry Point to Spark SQL and Spark DataFrame
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Description: this function is used for loading the song data from 
    AWS S3 bucket, processing the data and storing the songs, and 
    artists table back to S3 bucket.

    Args:
        spark: the spark session object
        input_data (string): the input data path
        output_data (string): the output data path

    Returns: 
        None
    """    
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
    
    # difine schema for song data dataframe
    stagingSongsSchema = R([
        Fld("song_id", Str()),
        Fld("num_songs", Int()),
        Fld("title", Str()),
        Fld("artist_name", Str()),
        Fld("artist_latitude", Dbl()),
        Fld("year", Int()),
        Fld("duration", Dbl()),
        Fld("artist_id", Str()),
        Fld("artist_longitude", Dbl()),
        Fld("artist_location", Str())
    ])

    # read song data file
    df = spark.read.json(song_data, schema=stagingSongsSchema)

    # create a temporary view for staging songs dataframe
    df.createOrReplaceTempView("staging_songs")

    # extract columns to create songs table
    songs_table = spark.sql(
        """
        SELECT 
            song_id,
            title,
            artist_id,
            year,
            duration
        FROM staging_songs
        """
    )
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode("ignore").partitionBy("year", "artist_id").parquet(output_data + "songs")

    # extract columns to create artists table
    artists_table = spark.sql(
        """
        SELECT
            artist_id,
            artist_name AS name,
            artist_location AS location,
            artist_latitude AS latitude,
            artist_longitude AS longitude
        FROM staging_songs
        """
    )
    
    # write artists table to parquet files
    artists_table.write.mode("ignore").parquet(output_data + "artists")


def process_log_data(spark, input_data, output_data):
    """
    Description: this function is used for loading log data from AWS S3 bucket,
    processing the data and storing the time, users, and songplays table back
    to S3 bucket.

    Args:
        spark: the spark session object
        input_data (string): the input data path
        output_data (string): the output data path

    Returns: 
        None
    """    
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # define schema for log dataframe
    stagingEventsSchema = R([
        Fld("artist", Str()),
        Fld("auth", Str()),
        Fld("firstName", Str()), 
        Fld("gender", Str()), 
        Fld("itemInSession", Int()), 
        Fld("lastName", Str()), 
        Fld("length", Dbl()), 
        Fld("level", Str()), 
        Fld("location", Str()), 
        Fld("method", Str()), 
        Fld("page", Str()), 
        Fld("registration", Str()), 
        Fld("sessionId", Str()), 
        Fld("song", Str()), 
        Fld("status", Str()), 
        Fld("ts", Str()), 
        Fld("userAgent", Str()), 
        Fld("userId", Str()) 
    ])

    # read log data file
    df = spark.read.json(log_data, schema=stagingEventsSchema)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # create a temporary view for staging events dataframe
    df.createOrReplaceTempView("staging_events")

    # extract columns for users table    
    users_table = spark.sql(
        """
        SELECT
            userId AS user_id,
            firstName AS first_name,
            lastName AS last_name,
            gender,
            level
        FROM staging_events
        """
    )
    
    # write users table to parquet files
    users_table.write.mode("ignore").parquet(output_data + "users")
    
    # create datetime column from original timestamp column
    start_time = spark.sql(
        """
        SELECT 
            from_unixtime(cast(ts as bigint)/1000,'yyyy-MM-dd HH:mm:ss') AS start_time
        FROM staging_events
        """
    ).dropDuplicates(['start_time'])
    
    # extract columns to create time table
    time_table = start_time.select(
        "start_time",
        hour("start_time").alias('hour'),
        dayofmonth("start_time").alias('day'),
        weekofyear("start_time").alias('week'),
        month("start_time").alias('month'),
        year("start_time").alias('year'),
        dayofweek("start_time").alias('weekday')
    )
    
    # write time table to parquet files partitioned by year and month
    time_table.write.mode("ignore").partitionBy('year', 'month').parquet(output_data + 'time')

    # read in song data to use for songplays table
    song_data = input_data + "song_data/*/*/*/*.json"
    stagingSongsSchema = R([
        Fld("song_id", Str()),
        Fld("num_songs", Int()),
        Fld("title", Str()),
        Fld("artist_name", Str()),
        Fld("artist_latitude", Dbl()),
        Fld("year", Int()),
        Fld("duration", Dbl()),
        Fld("artist_id", Str()),
        Fld("artist_longitude", Dbl()),
        Fld("artist_location", Str())
    ])
    song_df = spark.read.json(song_data, schema=stagingSongsSchema)
    song_df.createOrReplaceTempView("staging_songs")

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql(
        """
        WITH CTE AS (
            SELECT
                DISTINCT from_unixtime(cast(se.ts as bigint)/1000,'yyyy-MM-dd HH:mm:ss') AS start_time,
                se.userId, se.level, ss.song_id, ss.artist_id, se.sessionId AS session_id, se.location, se.userAgent AS user_agent
            FROM 
                staging_events se
            INNER JOIN 
                staging_songs ss
            ON 
                se.song = ss.title 
            AND 
                se.artist = ss.artist_name
        ) 
        SELECT 
            ROW_NUMBER() OVER(ORDER BY start_time) songplay_id,
            *, 
            YEAR(start_time) year, 
            MONTH(start_time) month 
        FROM 
            CTE
        """
    )

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode("ignore").partitionBy('year', 'month').parquet(output_data + 'songplays')


def main():
    """This function can be used to create a spark session, load song and log data
    from AWS S3, process the data, then create start schema, and finally store the
    fact and dimension tables to AWS S3 bucket.
    """    
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "./output/"
    
    # process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
