# shortest_path.py

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StructType, StructField
from pyspark.sql.functions import col, lit, when
import sys

def shortest_path(v_from, v_to, dataset_path=None, output_dir=None, max_path_length=100):
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("ShortestPathFinder") \
        .getOrCreate()
    
    # Define schema for graph edges
    graph_schema = StructType([
        StructField("user_id", IntegerType(), False),
        StructField("follower_id", IntegerType(), False)
    ])
    
    # Read graph edges from dataset
    edges = spark.read.csv(dataset_path, sep="\t", schema=graph_schema)       
    
    # Cache the edges DataFrame for better performance
    edges.cache()
    
    # Initialize DataFrame to store distances from the starting vertex
    dist_schema = StructType([
        StructField("vertex", IntegerType(), False),
        StructField("distance", IntegerType(), False)
    ])
    distances = spark.createDataFrame([(v_from, 0)], dist_schema)
    
    # Initialize variables for loop and distance counter
    d = 0
    while d < max_path_length:
        # Join distances DataFrame with edges DataFrame to find next vertices and their distances
        candidates = distances.join(edges, distances.vertex == edges.user_id)
        candidates = candidates.select(col("follower_id").alias("vertex"), (distances.distance + 1).alias("distance"))
        
        # Cache the candidates DataFrame for better performance
        candidates.cache()
        
        # Update distances DataFrame with new distances
        new_distances = distances.join(candidates, on="vertex", how="full_outer") \
            .select("vertex", 
                    when(distances.distance.isNull(), candidates.distance)
                    .when(candidates.distance.isNull(), distances.distance)
                    .otherwise(when(distances.distance < candidates.distance, distances.distance)
                               .otherwise(candidates.distance))
                    .alias("distance")) \
            .persist()
        
        # Count the number of vertices at distance d+1
        count = new_distances.where(new_distances.distance == d + 1).count()
        
        # Check if the target vertex has been reached
        target_reached = new_distances.where(new_distances.vertex == v_to).count() > 0
        
        # Break the loop if the target vertex has been reached or if there are no more vertices at distance d+1
        if target_reached or count == 0:
            break
        
        # Update distances DataFrame and distance counter
        distances = candidates
        d += 1
    
    # If the target vertex has been reached, extract and save the shortest paths
    if target_reached:
        shortest_paths = new_distances.where(new_distances.vertex == v_to).select("distance").distinct()
        shortest_paths.write.csv(output_dir, mode="overwrite", header=False)
    
    # Stop Spark session
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
