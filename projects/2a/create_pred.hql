drop table if exists hw2_pred;
create TABLE IF NOT EXISTS hw2_pred (id INT, pred FLOAT) 
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
WITH SERDEPROPERTIES('seporatorChar'='\t', 'quoteChar'='\"') STORED AS TEXTFILE 
 LOCATION 'Ega2901_hw2_pred';
