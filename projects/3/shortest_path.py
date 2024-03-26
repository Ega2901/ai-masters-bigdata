from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list
import sys

def bfs(graph, start_node, end_node):
    visited = set()
    queue = [[start_node]]

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node == end_node:
            return [path]

        if node not in visited:
            visited.add(node)
            neighbors = graph.filter(lambda edge: edge[0] == node).map(lambda edge: edge[1]).collect()

            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return []

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

    # Создаем RDD с данными графа
    edges = graph_data.rdd.map(lambda row: (row._1, row._2))

    # Выполняем алгоритм BFS для поиска кратчайшего пути
    shortest_paths = bfs(edges, start_node, end_node)

    # Сохраняем результаты в CSV файл
    shortest_paths.map(lambda path: ','.join(path)).coalesce(1).saveAsTextFile(output_dir)

    # Завершаем SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
