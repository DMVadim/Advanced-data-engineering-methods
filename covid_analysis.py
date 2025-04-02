from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window

# Создание SparkSession
spark = SparkSession.builder \
    .appName("COVID-19 Analysis") \
    .getOrCreate()

# Загрузка данных
df = spark.read.csv("/tmp/data/covid_data.csv", header=True, inferSchema=True)

# Задание 1: 15 стран с наибольшим процентом переболевших на 31 марта
df_march_31 = df.filter(col("date") == "2021-03-31")
df_march_31 = df_march_31.withColumn("percent_recovered", (col("total_cases") / col("population")) * 100)
top_15_recovered = df_march_31.select("iso_code", "location", "percent_recovered") \
    .orderBy(col("percent_recovered").desc()) \
    .limit(15)

top_15_recovered.show()

# Задание 2: Top 10 стран с максимальным количеством новых случаев за последнюю неделю марта 2021
df_march_last_week = df.filter((col("date") >= "2021-03-25") & (col("date") <= "2021-03-31"))
new_cases_summary = df_march_last_week.groupBy("location") \
    .agg({"new_cases": "sum"}) \
    .withColumnRenamed("sum(new_cases)", "total_new_cases")
top_10_new_cases = new_cases_summary.orderBy(col("total_new_cases").desc()).limit(10)

top_10_new_cases.show()

# Задание 3: Изменение случаев относительно предыдущего дня в России за последнюю неделю марта 2021
df_russia = df.filter(col("location") == "Russia") \
    .filter((col("date") >= "2021-03-25") & (col("date") <= "2021-03-31")) \
    .select("date", "new_cases")
window_spec = Window.orderBy("date")
df_russia = df_russia.withColumn("yesterday_new_cases", lag("new_cases").over(window_spec))
df_russia = df_russia.withColumn("delta", col("new_cases") - col("yesterday_new_cases"))

df_russia.show()