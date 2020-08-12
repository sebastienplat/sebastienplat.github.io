# INTRODUCTION

#### Load/Save Data

##### Read CSV

See also:

+ [pd.read_csv](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)
+ [df.to_csv](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html)

```
pd.read_csv('path_to_file', 
            index_col=None, 
            usecols=None, 
            dtype=None, 
            parse_dates=None # array of date columns
)
```

##### Save to CSV
```
# index=False: do not include the index in the csv
df.to_csv('path_to_file', index=False)
```

##### Empty DataFrame

```
df = pd.DataFrame(columns=[cols])
```

```
df = df.append(other_df, ignore_index=False, ...)
```



#### Dataframe Concepts

##### Index

[Reset index](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html)
[Set index](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.set_index.html)
```
df.reset_index(drop=False, inplace=False, ...)
```

See also the [documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html).


##### Multiindex

+ [get level values](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.MultiIndex.get_level_values.html)
+ [set levels](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.MultiIndex.set_levels.html)

##### loc, iloc, ix, xs


##### Rename rows index or columns

```
# use a dictionary
df.rename(
    index = {'old_name': 'new_name'}, 
    columns = {'old_name': 'new_name'}
)
```

See also: [docs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rename.html)


##### Rename index labels

```
# use an array that has the same length as the index
df.index = ['new_name1', 'new_name2', ...]
```



##### Drop columns

```
# use an array
df.drop(['Col1', 'Col2'], axis=1, level=None, inplace=False, errors=’raise’)
```

See also: [docs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)



#### Dataframe Manipulations

##### Sort

See also:

+ [sort_values API](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html)
+ [sort_index API](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_index.html)

```
df.sort_values(by, axis=0, ascending=True, inplace=False, ...)
df.sort_index(axis=0, ascending=True, inplace=False, na_position=’last’, ...)
```

##### Str Replace

See the [docs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.replace.html).


##### Series Apply

You can apply a function to all the elements of a Series. The first argument has to be the element value.

```
# no other arguments
def my_func(x):
	(...)
    
series.apply(my_func)
```

```
# additional positional arguments
def my_func(x, arg1, ...):
	(...)
    
series.apply(my_func, args=(arg1_value, ...))
```

```
# additional keyword arguments
def my_func(x, **args):
	(...)
    
series.apply(my_func, arg1, ...)
```

See also the [docs]().


##### Binning Values

See also the documentation on [pd.cut](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html).

When`retbins=True`, the bins values are also returned. It is an array of bins+1 values delimiting the bins intervals.

```python
# numpy array to bin
a = np.array([.2, 1.4, 2.5, 6.2])

# split values in three bins
out, bins = pd.cut(a, bins=3, retbins=True )

> [(0.194, 2.2] < (2.2, 4.2] < (4.2, 6.2]] # out categories
> array([ 0.194,  2.2  ,  4.2  ,  6.2  ]) # bins

# split values in three custom bins
out, bins = pd.cut(a, bins=[0, 1.5, 2.5, 7], retbins=True)

> [(0, 1.5] < (1.5, 2.5] < (2.5, 7]] # out categories
> array([ 0. ,  1.5,  2.5,  7. ]) # bins

# custom bins & labels
out, bins = pd.cut(a, bins=[0, 1.5, 2.5, 7], 
                   labels=['a','b','c'], retbins=True)

> [a < b < c] # out categories
> array([ 0. ,  1.5,  2.5,  7. ]) # bins
```



#### Grouping

##### Group By

[Apply multiple functions](http://pandas.pydata.org/pandas-docs/stable/groupby.html#applying-different-functions-to-dataframe-columns)

```
def my_func(x):
    col_1 = ...
    col_2 = ...
    return pd.Series([col_1, col_2], index=['col_1', 'col_2'])

# for each value of the column
grouped_df = df.groupby(['col']).apply(my_func)

# different functions
grouped_df = df.groupby(['col']).agg({'col_1': 'sum', 'col_2': 'count'})
```


##### Pivot Table

```
pd.pivot_table(df, index=None, columns=None, values=None, aggfunc='mean', ...)
```

See also the documentation on [pd.pivot\_table](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html).


##### Melt

> This function is useful to massage a DataFrame into a format where one or more columns are identifier variables (id_vars), while all other columns, considered measured variables (value_vars), are “unpivoted” to the row axis, leaving just two non-identifier columns, ‘variable’ and ‘value’.

```
pd.melt(df, 
    id_vars=[list], value_vars=[list], 
    var_name=None, value_name='value'
)
```

See also [melt](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html).



#### Combining DataFrames

See also the [documentation](http://pandas.pydata.org/pandas-docs/stable/merging.html).

+ [pd.concat](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html) appends along rows or columns, with optional set logic along the other ones (outer join by default)
+ [pd.merge](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.merge.html) performs a database-style join operation by columns or indexes
+ [df.join](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) performs a database-style join operation by indexes

```
pd.concat((df1, df2), axis=1, ...)
```

```
pd.merge(df1, df2, how='inner', on=None, left_on=None, right_on=None, ...)
```



#### Time Series

##### Dates Formatting

See also:

+ docs on [timeseries](http://pandas.pydata.org/pandas-docs/stable/timeseries.html)
+ [pd.to\_datetime API](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html)
+ [strftime.org](http://strftime.org/)


`pd.to_datetime` returns a datetime Series or DataFrame:

```
pd.to_datetime(series_to_convert, format='%Y-%m-%d %H:%M:%S', errors='coerce')
```

It is also possible to use `astype`:

```
series_to_convert = series_to_convert.astype('datetime64[ns]')
```


```
# datetime series - year / week / day
datetime_series.dt.year
datetime_series.dt.month
datetime_series.dt.weekofyear
datetime_series.dt.day
datetime_series.dt.dayofweek
datetime_series.dt.hour
```

See also the [complete list of date components](https://pandas-docs.github.io/pandas-docs-travis/timeseries.html#time-date-components).


```
# day of week as string
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
lmap = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
df['day_of_week'] = df['weekDay'].map(dmap)
df['day_of_week'] = pd.Categorical(df['day_of_week'], lmap)
```
Number of days between dates (in this example, a Pandas series minus a fixed date):

```
import datetime

# duration series
start_date = datetime.date(2011,1,24)
duration = df['end_date'].subtract(start_date)

# conversion into days as integer series
duration = duration.astype('timedelta64[D]').astype(int)
```

`datetime.replace` replace elements of a date:

```
import datetime
date = datetime.date(2007, 12, 7)
newdate = date.replace(year=2012)
print(date) #2007-12-07
print(newdate) #2012-12-07
```

See also the [docs](https://docs.python.org/3.5/library/datetime.html#datetime.datetime.replace).



##### Exploding Date Ranges

+ [pd.date\_range API](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html)

`pd.date_range` returns a datetime index of dates between a start and an end date:

```
import datetime
start_date = datetime.date(2007, 12, 5)
end_date = datetime.date(2007, 12, 7)
daterange = pd.date_range(start_date, end_date, freq=freq)
```

The `freq`parameter indicates how to split the date range; days by default.

This can be used to explode a date range into different rows (see this [SO question](https://stackoverflow.com/questions/42151886/expanding-pandas-data-frame-with-date-range-in-columns)):

```
pd.concat( 
  [
    pd.DataFrame(
      {
        'Date': pd.date_range(row.StartDateCol, row.EndDateCol, freq='D'),
        'col_1': row.col_1,
        'col_2': row.col_2
      }, 
      columns=['Date', 'col_1', 'col_2']
    ) 
    for i, row in df.iterrows()
  ], ignore_index=True
)
```




##### Lag Functions

See also: [lag with multiindex](http://stackoverflow.com/questions/38678246/use-pandas-dataframe-to-add-lag-feature-from-multiiindex-series)

See also [sort within groups](http://stackoverflow.com/questions/27842613/pandas-groupby-sort-within-groups)


#### Other useful commands

##### Split array into rows

[See this SO question](http://stackoverflow.com/questions/12680754/split-pandas-dataframe-string-entry-to-separate-rows)

See docs for [assign](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.assign.html).

##### Filter columns based on types

See also [filter columns based on their type](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html)

##### Convert String columns to Categorical Integers

```python
# convert string columns to categorical int
# with matching values in dictionary
label_mapping = {}
for col in X.dtypes[X.dtypes == 'object'].index:
    X[col], label_mapping[col] = pd.factorize(X[col])
```
3. [Series.value\_counts()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html): occurrences for each value
4. [Series.unique()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.unique.html): list of unique values
4. [Series.nunique()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.nunique.html) number of unique values


```
def func(row):
    if row['mobile'] == 'mobile':
        return 'mobile'
    elif row['tablet'] =='tablet':
        return 'tablet' 
    else:
        return 'other'

df['combo'] = df.apply(func, axis=1)
```




# 