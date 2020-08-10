# BIG DATA

#### Overview

The goal of collecting and processing volumes of complex data is to understand trends, uncover hidden patterns, detect anomalies, etc. By better understanding the problem being analyzed, organizations can make more informed, data-driven decisions:

+ **Personalized marketing**: purchases, center of interest, visited websites, etc.
+ **Recommendations engines** 
+ **Sentiment analysis** (opinion mining) in reviews
+ **Mobile advertising**: geolocalized ads, dicsounts, etc.
+ **Consumers behaviour** to guide products growth
+ **Biomedical applications**: genomics, personalized treatments, etc.
+ **Smart cities using real-time data** from city-wide sensors (traffic, pollution, etc.)


#### 3 V's

Big Data was born with the advent of cheap sensors, mobile phones and social media. 

+ **Volume**: Huge amount of data to process (storage / bandwidth / performance)
+ **Variety**: Mix of structured / unstructured data (tables / text / videos / ...)
+ **Velocity**: New data generated extremely frequently (real-time processing)


#### Process Steps

1. **Define purpose**: clarify the problem / define goal + success criteria
2. **Acquire data**: identify data sources / access + query data
3. **Explore**: understand the nature of data / correlations / trends
4. **Pre-process**: clean data (missing values / outliers / duplicates / etc.)
5. **Analyze**: filter + select features / build + evaluate model
6. **Communicate**: present results in a visual way, related to the success criteria
7. **Act**: implement + monitor actions based on insights gained from the analysis

Different types of model:

+ **Classification**: predict the category of the input data
+ **Regression**: predict a numeric value (stock price, etc.)
+ **Clustering**: organize similar items into groups (customer base into segments, etc.)
+ **Association Analysis**: market basket analysis for cross-selling, etc.
+ **Graph Analytics**: find connections between many entities (social networks, etc.)

Graph analysis can also be used to optimize mobile telecommunications network traffic for example.



# HADOOP AND BIG DATA

Big Data is a term used to describe large collections of data that grow so large and quickly that it is difficult to manage with regular databases or statistical tools. 

Hadoop is an open source project of the Apache Foundation. It is a framework written in Java, originally developed by Doug Cutting who named it after his son's toy elephant. 

Hadoop uses Google's MapReduce technology as its foundation. It is optimized to handle massive quantities of data using commodity hardware, ie relatively inexpensive computers.

This massive parallel processing is done with great performance. However, it is a batch operation handling massive amounts of data, so the response time is not immediate. 

Hadoop replicates its data across different computers, so that if one goes down, the data is processed on one of the replicated computers.

Hadoop is not suitable for OnLine Transaction Processing workloads, where data is randomly accessed on structured data like a relational database. Also, Hadoop is not suitable for OnLine Analytical Processing or Decision Support System workloads, where data is sequentially accessed on structured data like a relational database, to generate reports that provide business intelligence. 

As of Hadoop version 2.6, in place updates are not possible, but appends are possible. Hadoop is used for Big Data. It complements OnLine Transaction Processing and OnLine Analytical Processing. It is NOT a replacement for a relational database system. 

This is a list of some other open source project related to Hadoop:

- **Mapreduce** is a software framework for easily writing applications which processes vast amountsof data
- **Hive** provides data warehousing tools to extract, transform and load (ETL) data, and query this data stored in Hadoop files
- **Pig** is a high level language that generates MapReduce code to analyze large data sets
- ** ZooKeeper**  is a centralized configuration service and naming registry for large distributed systems
- **Yarn** is a large-scale operating system for big data applications
- **Spark** is a cluster computing framework 
- **Hbase** is a Hadoop database 
- **Eclipse** is a popular IDE donated by IBM to the open-source community
- **Lucene** is a text search engine library written in Java
- **Ambari** manages and monitors Hadoop clusters through an intuitive web UI
- **Avro**  is a data serialization system
- **UIMA**  is the architecture for the development, discovery, composition and deployment for the analysis of unstructured data

Areas where Hadoop is not good:

+ to process transactions due to its lack random access
+ when the work cannot be parallelized or when there are dependencies within the data, that is, record one must be processed before record two
+ for low latency data acces
+ for processing lots of small files although there is work being done in this area, for example, IBM's Adaptive MapReduce. 
+ for intensive calculations with little data. Now let's move on, and talk about Big Data solutions. 

Big Data solutions are more than just Hadoop. They can integrate analytic solutions to the mix to derive valuable information that can combine structured legacy data with new unstructured data. Big data solutions may also be used to derive information from data in motion, for example, IBM has a product called InfoSphere Streams that can be used to quickly determine customer sentiment for a new product based on Facebook or Twitter comments. 

Cloud computing has gained a tremendous track in the past few years, and it is a perfect fit for Big Data solutions. Using the cloud, a Hadoop cluster can be setup in minutes, on demand, and it can run for as long as needed without having to pay for more than what is used.