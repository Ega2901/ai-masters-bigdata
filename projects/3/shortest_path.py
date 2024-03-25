from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame
import sys

def main():
    # Проверяем количество аргументов
    if len(sys.argv) != 5:
        print("Usage: shortest_path.py <start_node> <end_node> <graph_path> <output_dir>")
        sys.exit(1)

    # Получаем аргументы из командной строки
    start_node = sys.argv[1]
    end_node = sys.argv[2]
    graph_path = sys.argv[3]
    output_dir = sys.argv[4]

    # Инициализируем SparkSession
    spark = SparkSession.builder \
        .appName("ShortestPath") \
        .getOrCreate()

    # Загружаем данные графа
    graph_data = spark.read.csv(graph_path, sep="\t", header=False)

    # Инвертируем направление ребер
    inverted_graph_data = graph_data.select(col("_c1").alias("src"), col("_c0").alias("dst"))

    # Разделяем данные на вершины и ребра
    vertices = graph_data.union(inverted_graph_data).selectExpr("_c0 as id").distinct()
    edges = inverted_graph_data

    # Создаем граф
    graph = GraphFrame(vertices, edges)

    # Выполняем BFS для поиска кратчайшего пути
    shortest_paths = graph.bfs(fromExpr=f"id = '{start_node}'", toExpr=f"id = '{end_node}'")

    # Сохраняем результаты в CSV файл
    shortest_paths.select("from.id", "to.id").write.csv(output_dir)

    spark.stop()

if __name__ == "__main__":
    main()
