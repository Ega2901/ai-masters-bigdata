CREATE TEMPORARY EXTERNAL TABLE IF NOT EXISTS  
hw2_test(
 id int,
 if1 double,
 if2 double,
 if3 double,
 if4 double,
 if5 double,
 if6 double,
 if7 double,
 if8 double,
 if9 double,
 if10 double,
 if11 double,
 if12 double,
 if13 double,
 cf1 string,
 cf2 string,
 cf3 string,
 cf4 string,
 cf5 string,
 cf6 string,
 cf7 string,
 cf8 string,
 cf9 string,
 cf10 string,
 cf11 string,
 cf12 string,
 cf13 string,
 cf14 string,
 cf15 string,
 cf16 string,
 cf17 string,
 cf18 string,
 cf19 string,
 cf20 string,
 cf21 string,
 cf22 string,
 cf23 string,
 cf24 string,
 cf25 string,
 cf26 string) 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY '\t' 
STORED AS TEXTFILE 
LOCATION '/datasets/criteo/testdir';
