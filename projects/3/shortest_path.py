from pyspark.sql import SparkSession
import sys

def bfs(graph, start_node, end_node, max_path_length):
    visited = set()
    queue = [[start_node]]

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if len(path) > max_path_length:
            continue

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
    edges = graph_data.rdd.map(lambda row: (row[0], row[1]))

    # Определяем максимальную длину пути как среднее значение длин всех путей
    avg_path_length = edges.map(lambda edge: (edge[0], len(edge[1]))).groupByKey().map(lambda x: (x[0], sum(x[1]) / len(x[1]))).collect()
    max_path_length = max(avg_path_length, key=lambda x: x[1])[1]

    # Выполняем алгоритм BFS для поиска кратчайшего пути с учетом максимальной длины
    shortest_paths = bfs(edges, start_node, end_node, max_path_length)

    # Сохраняем результаты в CSV файл
    spark.sparkContext.parallelize(shortest_paths).map(lambda path: ','.join(map(str, path))).coalesce(1).saveAsTextFile(output_dir)

    # Завершаем SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
