# SETUP

+ install package [rpy2](http://rpy2.readthedocs.io/en/default/overview.html#installation)
+ for Windows users, setup [environment variables](http://stackoverflow.com/questions/12698877/how-to-setup-enviroment-variable-r-user-to-use-rpy2-in-python)

On Windows, you will need to:

1. Add an environment variable `R_HOME` that points to the directory where R is installed
2. Add the path where you installed R libraries (Optional, if not default path)

```
import rpy2

import os
os.environ['R_HOME'] = 'C:\Program Files\R\R-3.2.3'
%R .libPaths('C:\\Users\\sebastien\\Documents\\R\\win-library\\3.2')

%load_ext rpy2.ipython
```

To transfer a variable from Python to R:

```
%R -i my_variable
```

To print properly on Windows, you need to use R function `capture_output` and pass it back to Python:

```
%R s <- capture.output({ my_variable })
a = %R s
for line in a:
    print(line)
```


# R ML

```
%%R

data <- subset(data, select = -c(a,c) )
data$feature = as.factor(data$feature)
```

```
%%R

train_sample = sample(nrow(data), size = nrow(data)*0.66)
train_data = data[train_sample,]
test_data = data[-train_sample,]

outcome = outcome_col_name

rf = randomForest(
    x = train_data[, names(train_data)!=outcome],
    y=train_data[, outcome], 
    xtest = test_data[, names(train_data)!=outcome],
    ytest = test_data[, outcome], 
    ntree = 10, mtry = 3, 
    keep.forest = TRUE, 
    classwt = c(0.7,0.3)
)
```

```
%R varImpPlot(rf,type=2)
```

```
%%R 

library(randomForest)

op <- par(mfrow=c(2, 2))
partialPlot(rf, train_data, country, 1)
partialPlot(rf, train_data, age, 1)
partialPlot(rf, train_data, new_user, 1)
partialPlot(rf, train_data, source, 1)
```