# shortest_path.py

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql.functions import col, lit, when, broadcast, concat_ws
import pyspark.sql.functions as f

conf = SparkConf()
sc = SparkContext(appName="Pagerank", conf=conf)

def shortest_path(v_from, v_to, dataset_path=None, output_dir=None):
    
    spark = SparkSession(sc)
    
    # Схема для графа и расстояний
    graph_schema = StructType([
        StructField("user_id", IntegerType(), False),
        StructField("follower_id", IntegerType(), False)
    ])
    dist_schema = StructType([
        StructField("vertex", IntegerType(), False),
        StructField("distance", IntegerType(), False)
    ])

    # Чтение данных из файла с репартиционированием
    edges = spark.read.csv(dataset_path, sep="\t", schema=graph_schema).repartition(10)

    # Кэширование
    edges_broadcast = f.broadcast(edges)

    # Начальные значения

    d = 0

    # Инициализация DataFrame для расстояний
    gr = spark.createDataFrame([(v_from, 0)], dist_schema).drop('distance')

    # Инициализация графа
    graph = f.broadcast(gr)

    cols = graph.columns


    while True:
        # Обновление расстояний
        distances = graph.select('vertex').distinct() \
                         .join(broadcast(edges_broadcast), edges_broadcast.follower_id == graph.vertex, "left") \
                         .filter(~f.col("user_id").isNull()) \
                         .drop('follower_id')

        # Проверка наличия целевой вершины
        if distances.filter(distances.user_id == v_to).count() > 0:
            # Построение графа для кратчайшего пути
            graph = distances.join(graph, on = "vertex", how = "left") \
                             .filter(f.col('user_id') == v_to) \
                             .withColumnRenamed('vertex', str(d)) \
                             .drop('user_id') \
                             .drop('vertex') 
            break
            
        if d < 2:
            graph = distances.join(graph, on = "vertex", how = "left") \
                         .filter((f.col("user_id") != graph.vertex) & (f.col("user_id") != graph[d])) \
                         .withColumnRenamed('vertex', str(d)) \
                         .withColumnRenamed("user_id", 'vertex') \
        
        else:
            graph = distances.join(graph, on = "vertex", how = "left") \
                             .filter((f.col("user_id") != graph.vertex) & (f.col("user_id") != graph[d]) & (f.col("user_id") != graph[d-1]) & (f.col("user_id") != graph[d-2])) \
                             .withColumnRenamed('vertex', str(d)) \
                             .withColumnRenamed("user_id", 'vertex') \

            
        cols = graph.columns

        # Обновление расстояний
        d += 1

    # Вывод кратчайшего пути
    shortest_paths = graph
    cols = graph.columns
    rcols = cols[::-1]
    shortest_paths = graph.select(concat_ws(",", *rcols, lit(str(v_to))).alias("path"))
    
    shortest_paths.show.write.csv(output_dir, mode="overwrite", sep="\n", header=False)

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: shortest_path.py <start_vertex> <end_vertex> <dataset_path> <output_dir>")
        sys.exit(1)

    start_vertex = int(sys.argv[1])
    end_vertex = int(sys.argv[2])
    dataset_path = sys.argv[3]
    output_dir = sys.argv[4]

    shortest_path(start_vertex, end_vertex, dataset_path, output_dir)
