_Note: SQL stands for **S**tructured **Q**uery **L**anguage_

# RDBMS

#### Terminology

**RDBMS** stands for Relational DataBase Management System. It is a DBMS based on the relational model, introduced in the early 1970s by IBM.

In a  [RDBMS](https://en.wikipedia.org/wiki/Relational_database), data is stored in tables (or "relations") of columns and rows. 

+ each table represents one "entity type" (such as customer or product)
+ each row represents an instance of that entity type (such as "Lee" or "chair") 
+ columns represents the values attributed to that instance (such as address or price)

_Note: rows are also called "records" or "tuples". Columns are also known as "fields"._

**Relationships** are a logical connection between different tables, established on the basis of interaction among these tables.


#### Constraints

Constraints are rules enforced on data columns or tables. They are used to limit the type of data that can go into a table. This ensures the accuracy and reliability of the data in the database.

There are several commonly used SQL constraints:

##### Primary Key

The **PRIMARY KEY** is a field or group of fields (in that case, we talk about composite key) that uniquely identifies each row of the table:

+ no record can have a NULL primary key value
+ no two records can have the same primary key value


##### Secondary Key

The **FOREIGN KEY** is a field or group of fields (in that case, we talk about composite key) whose values match a Primary Key in a different table.

The purpose of the foreign key is to ensure referential integrity of the data. In other words, only values that are supposed to appear in the database are permitted. Operations that violate the referential integrity of this relationship will return an error. 

For example, let's consider two tables: CUSTOMER and ORDERS. Business logic requires that all orders are associated with an existing customer. To enforce this logic, we can place a foreign key on the ORDERS table that references the primary key of the CUSTOMER table. With it, the ORDERS table cannot contain information on a customer that is not in the CUSTOMER table.

+ inserting a row in ORDERS with a customer ID not in CUSTOMERS
+ deleting a row in CUSTOMERS whose ID is in the ORDERS foreign key column

##### Other constraints

+ **UNIQUE**: ensures that all values in a column are different
+ **NOT NULL**: ensures that a column cannot have NULL value
+ **DEFAULT**: provides a default value for a column when none is specified
+ **CHECK**: ensures that all values in a column satisfy certain conditions


#### Data Integrity

The following categories of the data integrity exist with each RDBMS:

+ **Entity Integrity**: ensures no duplicate rows in a table
+ **Domain Integrity**: enforces valid entries for a given column by restricting the type, the format, or the range of values
+ **Referential integrity**: rows cannot be deleted, which are used by other records
+ **User-Defined Integrity**: enforces some specific business rules that do not fall into entity, domain or referential integrity


#### Normalization

[Database normalization](https://en.wikipedia.org/wiki/Database_normalization) is the process of efficiently organizing data in a database. There are two reasons of the normalization process:

+ Eliminating redundant data
+ Ensuring data dependencies make sense

Normalization consists of a series of guidelines divided into successive normal forms. The [first three](http://phlonx.com/resources/nf3) are the most commonly used. 

1. No repeating elements or groups of elements
2. No partial dependencies on a concatenated key
3. No dependencies on non-key attributes

##### 1NF

No Repeating Elements or Groups of Elements (atomicity).

+ Create a separate table for each set of related data
+ Identify each set of related data with a primary key
+ Store data in atomic (indivisible) values (ie. one atomic value per row per column)


##### 2NF

No Partial Dependencies on a Composite Key. 

Each column in the table that is not part of the primary key must depend upon the entire concatenated key for its existence.

For each column, we ask this question: Can this column exist without one or the other part of the concatenated primary key?

If the answer is "yes" — even once — then the table fails Second Normal Form and we must create another table to rectify the failure.


##### 3NF

No Dependencies on Non-Key Attributes.

The third norm was designed to improve database processing while minimizing storage costs:

+ reduces the duplication of data
+ ensures referential integrity by having one unique source for each value (so any update is automatically applied to the whole database)

For each column, we ask this question: Can it exist independent of the primary key?

If the answer is "yes" — even once — then the table fails Third Normal Form and we must create another table to rectify the failure.


#### ACID transactions

[ACID](https://en.wikipedia.org/wiki/ACID) stands for Atomicity, Consistency, Isolation, and Durability.

+ **Atomicity**: requires that each transaction be "all or nothing". If one part of the transaction fails, then the entire transaction fails, and the database state is left unchanged.
+ **Consistency**: ensures that any transaction will bring the database from one valid state to another, according to all defined rules (constraints, cascades, triggers, ...)
+ **Isolation**: ensures that the concurrent execution of transactions results in a system state that would be obtained if transactions were executed sequentially, i.e., one after the other. 
+ **Durability**: ensures that once a transaction has been committed, it will remain so even in the event of power loss, crashes, or errors




# QUERIES

#### Clauses Order

SQL clauses are executed in the following order:

```
FROM, JOINs   -- determine & filter rows
WHERE         -- more filters on the rows; AND / OR / IN (., .) / LIKE '%_.'
GROUP BY      -- combine those rows into groups
HAVING        -- filter groups
SELECT        -- filter columns
DISTINCT
ORDER BY      -- arranges the remaining rows/groups; ASC / DESC
LIMIT         -- limits the results
```

#### SELECT

**Filter columns** to return.

```sql
-- returns all the table columns
SELECT * FROM myTable;

-- returns the selected columns
SELECT col1, col2 FROM myTable;

-- alias
SELECT col1, col2 AS col2_alias FROM myTable;
```

_Note: `SELECT` is executed after the `WHERE` and `GROUP BY`, so aliases might not work for these two clauses, depending on the DB used. See 'Subqueries' below for an example of how to handle it._

##### SELECT DISTINCT

`SELECT DISTINCT` can be used with aggregate functions:

```
SELECT COUNT (DISTINCT col1)
  FROM myTable
```

##### OVER (PARTITION BY)

`OVER` creates windows within a query. Each window expose a limited set of rows to an aggregation function, that is applied to each of its rows. 

Each window is treated separately from each other. 

`OVER` is an efficient way to compute aggregated values such as moving averages, cumulative aggregates, running totals, or a top N per group results.

It is also possible to apply window functions like `ROW_NUMBER()` (see example below).

A common way to create windows is to use `PARTITION BY col1`. Each window can then be sorted via `ORDER BY`. 

The  example below uses `OVER (PARTITION BY ...)` to identify the first user created for each company of the table `users`:

```
SELECT 
  company_id,
  user_id,
  created_at
FROM (
  SELECT 
    company_id,
    user_id,
    created_at,
    ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY created_at) 
    AS rownumber
  FROM users
) new_table
WHERE rownumber = 1;
```

```
SELECT
  user_id,
  occurred_at,
  LAG(occurred_at, 1) 
    OVER (PARTITION BY user_id ORDER BY occurred_at) 
    AS previous
FROM q1;
```

For each user, find the difference between the last action and the second last action.

```
SELECT
  user_id,
  diff
FROM (
  SELECT 
    user_id,
    occurred_at - previous AS diff,
    ROW_NUMBER() 
      OVER (PARTITION BY user_id ORDER BY occurred_at DESC)
      AS rownumber
  FROM (
    SELECT
      user_id,
      occurred_at,
      LAG(occurred_at, 1) 
       OVER (PARTITION BY user_id ORDER BY occurred_at) 
       AS previous
   FROM q1
  ) t_lag
  WHERE previous IS NOT NULL
) t_diff
WHERE rownumber = 1
```




#### WHERE

**Filter rows** to take into account. Possible to use aliases.

```sql
-- AND / OR / IN (., .) / LIKE '%_.'
-- %: zero or more characters
-- _: exactly one character
```

```sql
-- filter rows; WHERE should be placed before ORDER BY
SELECT col1 FROM myTable WHERE col1 > 10 ORDER BY col2;

-- AND, OR operators
SELECT col1 FROM myTable WHERE (col1 > 10 AND col1 < 20) OR col2 != 0;

-- IN operator (check if value belongs to a list; aggregation of OR operators)
SELECT col1 FROM myTable WHERE col1 IN ( value1, value2, ... );
SELECT col1 FROM myTable WHERE col1 NOT IN ( value1, value2, ... );

-- LIKE operator (search for specific patterns; % and _ as wilcards)
SELECT col1 FROM myTable WHERE col1 LIKE "%myKeyword%";

-- REGEXP
SELECT col1 FROM myTable WHERE (col2 REGEXP 'myRegexp');
```

_Note1: `AND` has precedence over `OR`._

_Note2: Regexp calls have a database-dependant syntax. The example above is for MySQL._

#### GROUP BY

```sql
-- MIN / MAX / SUM / AVG
-- COUNT (column): number of values in a column, excluding NULL
-- COUNT  (*): number of rows in the table
```

The `GROUP BY` operation returns only **one row per group**. So the `SELECT` clause, that occurs afterwards, should only include columns that have a unique value per group (aggregate functions). 

```sql
SELECT col1, SUM(col2) FROM myTable GROUP BY col1;
```

#### HAVING

**Filter grouped rows**. Possible to use aliases. Original columns are not accessible.

```sql
SELECT col1, SUM(col2) AS sum_col2 FROM myTable 
  GROUP BY col1
  HAVING sum_col2 > 100
  ;
  
-- Filter by occurrences count
SELECT col1 FROM myTable 
  GROUP BY col1
  HAVING COUNT(*) >= 10
  ;
```

#### CASE

**Create new column** whose values are created from conditions on other columns. Can then be used with `GROUP BY`.

```sql
-- returns the number of rows in different ranges of col2
SELECT 
  COUNT(*), 
  CASE
    WHEN col2 > 200 THEN 'over 200'
    WHEN col2 > 100 THEN 'between 100 and 200'
    ELSE 'below 100'
  END 
  AS new_column
FROM myTable
GROUP BY new_column;
```

#### ORDER BY

**Sort by column(s)**.

```sql
-- ASC / DESC
```

```sql
-- order by column (the ordered column don't have to be selected)
SELECT col1 FROM myTable ORDER BY col2;
SELECT col1 FROM myTable ORDER BY col2 ASC;  -- ascending order, default
SELECT col1 FROM myTable ORDER BY col2 DESC; -- descending order

-- order by multiple columns
-- (ASC is the default for each column independantly)
SELECT col1 FROM myTable ORDER BY col1, col2;
SELECT col1 FROM myTable ORDER BY col1, col2 DESC;      -- only col2 is DESC
SELECT col1 FROM myTable ORDER BY col1 DESC, col2 DESC; -- both cols are DESC
```

#### Convert String Type

Types can be converted as follows:

```sql
-- in the SELECT clause
SELECT CAST(MyVarcharCol AS INT) FROM Table;

-- in the WHERE clause
SELECT * FROM Table WHERE CAST(MyVarcharCol AS INT) > 10;
```

#### Subqueries

It is possible to create queries inside queries to build intermediate tables. For example: 

```sql
-- init new table
CREATE TABLE student_grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    number_grade INTEGER,
    fraction_completed REAL);
  
-- add entries
INSERT INTO student_grades (name, number_grade, fraction_completed)
    VALUES ("Winston", 90, 0.805);
INSERT INTO student_grades (name, number_grade, fraction_completed)
    VALUES ("Winnefer", 95, 0.901);
INSERT INTO student_grades (name, number_grade, fraction_completed)
    VALUES ("Winsteen", 85, 0.906);
INSERT INTO student_grades (name, number_grade, fraction_completed)
    VALUES ("Wincifer", 66, 0.7054);
INSERT INTO student_grades (name, number_grade, fraction_completed)
    VALUES ("Winster", 76, 0.5013);
INSERT INTO student_grades (name, number_grade, fraction_completed)
    VALUES ("Winstonia", 82, 0.9045);
```
 
```sql
-- example of subquery usage
SELECT
  COUNT(*),
  CASE
    WHEN percent_completed > 90 THEN 'A'
    WHEN percent_completed > 80 THEN 'B'
    WHEN percent_completed > 70 THEN 'C'
    ELSE 'F'
  END
  AS letter_grade
FROM
  -- sub query to create new column 'percent_completed'
  (
  SELECT 
    name, 
    number_grade, 
    ROUND(100*fraction_completed) AS percent_completed
  FROM student_grades
  )
  AS new_table
  -- end of subquery
GROUP BY letter_grade;
```

The result:

<div class="row">
<div class="col-md-6">
!!
| COUNT(*)	| letter_grade | 
| :---: | :---: |
| 1	| A | 
| 3 | B | 
| 1	| C | 
| 1	| F |
!!
</div>
</div>




# JOINS

#### Purpose of Join

Merge two tables into one, to return attributes from both. Records are matched using keys.


#### Join Types

The joined tables might not have the same list of unique keys. To handle all the desired outputs, there are four types of joins:

<div class="row">

<div class="col-sm-6">
<img alt='inner-join' class="center-block" src='https://sebastienplat.s3.amazonaws.com/352466a378a553a0031a5eb14860cc221479381469558'/>
<p class="text-center">Select all records with matching keys.</p>
</div>

<div class="col-sm-6">
<img alt='left-join' class="center-block" src='https://sebastienplat.s3.amazonaws.com/2a5d3ecda7d5173c139194f0118ba8911479381479478'/>
<p class="text-center">Select all records from A. Attributes from B are `NULL` for records without matching key.</p>
</div>

<div class="clearfix"></div>

<div class="col-sm-6">
<img alt='right-join' class="center-block" src='https://sebastienplat.s3.amazonaws.com/9b0417486a071b81455df1f66240ac441479381487513'/>
<p class="text-center">Select all records from B. Attributes from A are `NULL` for records without matching key.</p>
</div>

<div class="col-sm-6">
<img alt='full-join' class="center-block" src='https://sebastienplat.s3.amazonaws.com/205fff986a1b623e83b5954fe8150ff01479381494620'/>
<p class="text-center">Select all records from both tables. Attributes are `NULL` for records without matching key.</p>
</div>

</div>

*source: [sql-join.com](http://www.sql-join.com/sql-join-types)*

*Note: `A LEFT JOIN B` is the same as `B RIGHT JOIN A`.*

*Note: `LEFT JOIN`, `RIGHT JOIN` and `FULL JOIN` are called `LEFT OUTER JOIN`, `RIGHT OUTER JOIN` and `FULL OUTER JOIN` in some databases.*


#### JOIN

```sql
-- joining two tables
SELECT 
  table1.col1,
  table1.col2,
  table2.col1
FROM 
  table1
JOIN / LEFT OUTER JOIN / RIGHT OUTER JOIN / FULL OUTER JOIN
  table2
ON 
  table1.id = table2.table1_id
;
```

```sql
-- self join
CREATE TABLE movies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  released INTEGER,
  sequel_id INTEGER
);
    
SELECT
  movies.title,
  sequels.title AS sequel_title
FROM 
  movies
LEFT OUTER JOIN 
  movies AS sequels
ON 
  movies.sequel_id = sequels.id;
```

```sql
-- successive joins
CREATE TABLE persons (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fullname TEXT,
  age INTEGER
);

CREATE table friends (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person1_id INTEGER,
  person2_id INTEGER
);

SELECT 
  friends_1.fullname,
  friends_2.fullname
FROM 
  friends
JOIN 
  persons AS friends_1
ON 
  friends_1.id = friends.person1_id
JOIN 
  persons AS friends_2
ON
  friends_2.id = friends.person2_id;
```




# DATES

It is very common to convert date string into a format that can be used by a computer.

```
SELECT
  FROM_ UNIXTIME (UNIX_TIMESTAMP(col_name), 'format')
FROM my_table;
```

Format describes the way the date string is formatted. As an example:

```
2014-12-20 12:01:34 # example date
yyyy-MM-dd hh:mm:ss # corresponding format
```

We can then keep only the date like so:

```
SELECT
  CAST(SUBSTR(
    FROM_UNIXTIME (UNIX_TIMESTAMP(col_name), 
                    'yyyy-MM-dd hh:mm:ss'), 1, 10) AS DATE) 
  AS event_date  
FROM my_table;
```




# MODIFYING TABLES

#### CREATE TABLE

```sql
-- primary key: row identifier (must have unique values)
CREATE TABLE myTable ( id INTEGER PRIMARY KEY, col1 TEXT, col2 INTEGER, ...);

-- insert: values listed in the order we declared them
INSERT INTO myTable VALUES ( id_value, col1_value, col2_value, ... );
```
To make items creation easier, it is possible to let the database create `id` values automatically using `AUTOINCREMENT`.

```sql
-- autoincrement id
CREATE TABLE myTable ( id INTEGER PRIMARY KEY AUTOINCREMENT, ... )

-- insert values for only some fields of the table
INSERT INTO myTable( col1, col2 ) VALUES ( col1_value, col2_value );

-- missing values: col1 will be NULL
INSERT INTO myTable( col2 ) VALUES ( col1_value );
```

_Note: missing values will be replaced by `NULL` for all columns without `AUTOINCREMENT`._


#### UPDATE & DELETE

```sql
-- UPDATE
UPDATE myTable 
  SET col1 = newValue 
  WHERE col2 = condition;

-- DELETE
DELETE FROM myTable 
  WHERE col1 = condition;
```

*Note: `UPDATE` and `DELETE` are irreversible.*


#### ALTER & DROP TABLE

```sql
-- ADD: name of new column plus its type & default value
ALTER TABLE myTable ADD myNewColumn TEXT default "unknown";

-- DROP TABLE
DROP TABLE myTable;
```


