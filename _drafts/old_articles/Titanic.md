This article takes insights from this excellent [tutorial from Trevor Stephens](http://trevorstephens.com/kaggle-titanic-tutorial/getting-started-with-r/). Information regarding the RMS Titanic come from [Wikipedia](https://en.wikipedia.org/wiki/RMS_Titanic) and  [The Encyclopedia Titanica](https://www.encyclopedia-titanica.org).

# GETTING STARTED

#### Introduction

Before diving into Kaggle competitions, let's work on [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic). It seems like an excellent starting point, as the dataset only has a few, easily understandable, variables.

> The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

![titanic.png](https://upload.wikimedia.org/wikipedia/commons/5/51/Titanic_voyage_map.png)

There were not enough lifeboats to accommodate all of those aboard, as their total capacity was only 1200 people. In addition, many of the lifeboats were launched only partially loaded, which explains the high number of lost lives. 

We have all heard about the ["women and children first"](https://en.wikipedia.org/wiki/Women_and_children_first) protocol for loading lifeboats. We will see how this can help us with our first predictions.


#### Available Data

As in most Kaggle competitions, we are given a training set and a test set.

+ **training set**, complete with the outcome (target variable): the dataset we will use to train our model
+ **test set** for which we must predict the now unknown target variable, based on our model

Both sets have the same variables, except for the target. Let's start by merging the two sets for our exploratory analysis:

```r
train <- read.csv("train.csv")
test <- read.csv("test.csv")

test$Survived <- NA
combi <- rbind(train, test)
```

Let's have a look at the combined set:

```r
> str(combi)

'data.frame':	1309 obs. of  12 variables:
 $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
 $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
 $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
 $ Name       : Factor w/ 1307 levels "Abbing, Mr. Anthony",..: 109 191 358 277 16 559 520 629 417 581 ...
 $ Sex        : Factor w/ 2 levels "female","male": 2 1 1 1 2 2 2 2 1 1 ...
 $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
 $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
 $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
 $ Ticket     : Factor w/ 929 levels "110152","110413",..: 524 597 670 50 473 276 86 396 345 133 ...
 $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
 $ Cabin      : Factor w/ 187 levels "","A10","A14",..: 1 83 1 57 1 1 131 1 1 1 ...
 $ Embarked   : Factor w/ 4 levels "","C","Q","S": 4 2 4 4 4 3 4 4 4 2 ...
```

There are information about all 1,300 passengers.


#### Exploratory Analysis

> The most important step of any Machine Learning project is to make the most of the available data, a task called feature engineering *(see [this article](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) for more insights)*. Knowing the topic can be very helpful.


##### Pclass

The RMS Titanic had three classes for passengers, which is indicated in the dataset as `Pclass`:

```
> table(combi$Pclass)

  1   2   3 
323 277 709 
```


##### Name

We have the name of all the 1309 passengers. Let's have a closer look:

```r
> head(combi$Name)

Braund, Mr. Owen Harris
Cumings, Mrs. John Bradley (Florence Briggs Thayer)
Heikkinen, Miss. Laina
Futrelle, Mrs. Jacques Heath (Lily May Peel)
Allen, Mr. William Henry
Moran, Mr. James
```

It looks like the names all follow the same pattern: 

1. `Last Name`
2. `Title`
3. `First name(s)`

Having a closer look at the Title might prove useful.


##### Age

The age information is missing for roughly 20% of all passengers:

```r
> table(is.na(combi$Age))

FALSE  TRUE 
 1046   263 
```

As we want to identify children, we will need to estimate the missing values


##### Family & Ticket

`SibSp`and `Parch` stand for **Sib**lings/**Sp**ouse and **Par**ents/**Ch**ildren, respectively.

```
> table(combi$SibSp)

  0   1   2   3   4   5   8 
891 319  42  20  22   6   9 

> table(combi$Parch)

   0    1    2    3    4    5    6    9 
1002  170  113    8    6    6    2    2 
```

It appears most people travelled alone, but looking at families as a group might be interesting. We can assume that people would not want to get separated from loved ones, which could have been problematic for large groups.

There are 929 tickets for the 1309 passengers (929 factors in the variable), which means several passengers for one ticket. We can try to use them, combined with the family information, to identify groups of people travelling together. 


##### Fare

According to [Wikipedia](https://en.wikipedia.org/wiki/RMS_Titanic#Passengers), Third Class fares from London, Southampton, or Queenstown cost £7.50, while the cheapest First Class fares cost £23. The most expensive First Class suites were to have cost up to £870 in high season.

Some passengers 

 ##### Cabin
 
We can assume that fares are related to the position of the cabin in the ship. Due to the late hour Titanic struck the iceberg, passengers were probably mostly in their quarters.

Looking at [ship's layout](http://www.copperas.com/titanic/), cabins were organized as follows:

| Deck | Fore (front) | Amidships (middle) | Aft (back) |
|:--------|:-------:|:----------------:|:-----:|
| Boat                    | - | - | - |
| Promenade (A) | - | Cabins (1st) | - |
| Bridge (B)          | - | Suites, cabins (1st) | - |
| Shelter (C)         | - | Cabins (1st) | - |
| Saloon (D)         | - | Cabins (1st) | Cabins (2nd+3rd) |
| Upper (E)           | Cabins (3rd) | Cabins (1st+2nd) | Cabins (2nd+3rd) |
| Middle (F)          | Cabins (3rd) | - | Cabins (2nd+3rd) |
| Lower (G)          | Cabins (3rd)  | - | Cabins (3rd)  |
| Orlop                 | - | - | - |
| Tank Top           | - | - | - |

First- and Second-Class were much closer to the boats deck than Third-Class accomodations, situated on the far ends of decks C to G. In addition, barriers and partitions segregated the accommodation for the steerage passengers from the others *([source](https://en.wikipedia.org/wiki/Sinking_of_the_RMS_Titanic#Departure_of_the_lifeboats_.2800:45.E2.80.9302:05.29))*. As Irish survivor Margaret Murphy wrote in May 1912:

> Before all the steerage passengers had even a chance of their lives, the Titanic's sailors fastened the doors and companionways leading up from the third-class section. [...] It meant all hope was gone for those still down there.

But information about cabins is scarce:

```r
> table(combi$Cabin != "")

FALSE  TRUE 
 1014   295 
```

We can try to combine this with family information to extrapolate the cabin deck of all passengers: families probably shared cabins. 

_Note: the fare is missing for one of the passengers; we will have to add it before using random forests._

##### Embarked

Finally, we have information about the embarkation port. The RMS Titanic left Southampton on Wednesday 10 April 1912 at noon. It reached Cherbourg later the same day, then picked its last passengers at Queenstown, Ireland the next day.

```r
> table(combi$Embarked, combi$Pclass)
   
      1   2   3
      2   0   0
  C 141  28 101
  Q   3   7 113
  S 177 242 495
```

Most passengers that embarked at Queenstown were travelling Third-Class. Information is missing for two passengers, so we will assume they embarked at Southampton. 

It doesn't seem likely that the port of embarkation played any role in a passenger's survival, except for the correlation between it and their class.


#### Using External Information

The Kaggle Leaderboard show a few submissions with 100% prediction accuracy. No Machine Learning algorithm could match this, but it illustrates an interesting point: access to external information.

This [Wikipedia page](https://en.wikipedia.org/wiki/Passengers_of_the_RMS_Titanic) lists all passengers of the RMS Titanic, including their name and whether they survived or not. Using this list, it is trivial to accurately "predict" the fate of each person in the test set.




# FEATURES ENGINEERING

#### Title

The name of each passenger  follows the same pattern: First Name `comma space` Title `point` Last Name.

We can use regex to extract their title:

```r
combi$Name <- as.character(combi$Name)
combi$Title <- sapply(combi$Name, function(x) {
  gsub("(^.+, )([A-Za-z ]+)(\\..+$)", "\\2", x)
})
```

The regex is not trivial. It looks for a pattern divided in three chunks:

+ `(^.+, )`: all characters from the start until, and including, `comma space`
+ `([A-Za-z ]+)`: all letters and `space` after chunk N°1 and before chunk N°3
+ `(\\..+$)`: all characters from `point`to the end

`\\2` indicates that we want what is inside the second chunk: the passenger's title.

Here are the results:

```r
> table(combi$Title)

        Capt          Col          Don         Dona           Dr     Jonkheer 
           1            4            1            1            8            1 
        Lady        Major       Master         Miss         Mlle          Mme 
           1            2           61          260            2            1 
          Mr          Mrs           Ms          Rev          Sir the Countess 
         757          197            2            8            1            1 
```

Some titles have only a few occurrences, which will be difficult for our model to use efficiently. Let's simplify:

```r
# Female titles
> combi$Title <- sub('Mme','Mrs', combi$Title)
> combi$Title <- sub('Mlle|Ms','Miss', combi$Title)

# Nobility/Honorific titles
> combi$Title <- sub('Dona|the Countess','Lady', combi$Title)
> combi$Title <- sub('Capt|Col|Don|Jonkheer|Major','Sir', combi$Title)

# as factor
combi$Title <- as.factor(combi$Title)
```

This gives us a much cleaner Title list:

```r
table(combi$Title)

    Dr   Lady Master   Miss     Mr    Mrs    Rev    Sir 
     8      3     61    264    757    198      8     10
```

_Note: 'Master' seems to designate young males, regardless of their travelling Class._


#### Group Size

People probably wanted to stay close to their loved ones, either family or friends, during this tragedy. We will estimate each group size by looking at:

+ the size of each passenger's family: their siblings, spouses, children & parents + themselves
+ the number of people sharing the same ticket (we will call it the ticket size)

Families could have bought their tickets separately, but it seems unlikely that people sharing the same ticket are not related.

We will compare the family size of each passenger with their ticket size. There are three possibilities:

1. **Family Size = Ticket Size**: the family has booked the trip as one. We will use the ticket N° as Group ID, and the ticket size as Group Size
2.  **Family Size < Ticket Size**: several friends travel together. We will use the ticket N° as Group ID, and the Ticket Size as Group Size
2.  **Family Size > Ticket Size**: several tickets for the family. We will use the last name as Group ID, and the Family Size as Group Size.

_Note: the Group ID will be use for the new Deck variable. See below._

```r
# family size
combi$FamilySize <- combi$SibSp + combi$Parch + 1

# passengers per ticket
TicketSize <- data.frame(table(combi$Ticket))
names(TicketSize) <- c("Ticket", "TicketSize")
combi <- merge(x = combi, y = TicketSize, by = "Ticket", all = TRUE)

# create groups & groups size
combi$groupID <- as.character(combi$Ticket)
combi$groupSize <- combi$TicketSize

# large families
combi$LastName <- sapply(combi$Name, function(x) {strsplit(x, ",")[[1]][1]})
largeFamilies <- combi$FamilySize > combi$TicketSize
combi$groupID[largeFamilies] <- as.character(combi$LastName[largeFamilies])
combi$groupSize[largeFamilies] <- combi$FamilySize[largeFamilies]
```

There are 920 groups of passengers, most of which are either couples or passengers travelling alone:

```r
# number of groups for each group size
> table(data.frame(table(combi$groupID))$Freq)

  1   2   3   4   5   6   7   8  11 
699 133  53  16   7   4   5   2   1 
```




# MISSING VALUES

#### Fare & Embarked

There are very few missing values for Fare and Embarked: one and two respectively. 

The passenger with missing fare was travelling third class, so we will just take the median:

```
combi$Fare[is.na(combi$Fare)] <- mean(combi$Fare[combi$Pclass==3], na.rm=TRUE)
```

The main Embarkation port was Southampton, so we will use `S` to replace the missing Embarked information:

```
combi$Embarked[combi$Embarked == ""] <- "S"
combi$Embarked <- droplevels(combi$Embarked) # dropping `NA`level
```
 

#### Age

A closer look at missing ages reveals, perhaps unsurprisingly, that it occurs mostly for third-class passengers:

```
table(combi$Sex, combi$Pclass, is.na(combi$Age))

           1   2   3
  female  11   3  64
  male    28  13 144
```

Age distribution are slightly skewed, so we will use `median` instead of `mean` to fill missing values: 

```
p <- qplot(Age, data = combi, geom = "histogram", binwidth = 10)
p + facet_wrap(Sex ~ Pclass)
```
![agePlot.png](https://sebastienplat.s3.amazonaws.com/35622d4a10eb2655312e585be37923921478167384628)

We also use the Title feature to get better estimates ("Master" indicates children, "Misses" are younger that "Mrs", etc.).

```
# Ages median
ages <- with(combi, tapply(Age, list(Sex, Pclass, Title), median, na.rm = TRUE))
ages <- data.frame(ftable(ages))
ages <- ages[complete.cases(ages), ]
colnames(ages) <- c('Sex','Pclass','Title','Age')

ages

      Sex Pclass  Title  Age
1  female      1     Dr 49.0
2    male      1     Dr 47.0
4    male      2     Dr 38.5
7  female      1   Lady 39.0
14   male      1 Master  6.0
16   male      2 Master  2.0
18   male      3 Master  6.0
19 female      1   Miss 30.0
21 female      2   Miss 20.0
23 female      3   Miss 18.0
26   male      1     Mr 41.5
28   male      2     Mr 30.0
30   male      3     Mr 26.0
31 female      1    Mrs 45.0
33 female      2    Mrs 30.5
35 female      3    Mrs 31.0
40   male      2    Rev 41.5
44   male      1    Sir 50.5
```

Now that we have are medians, we can apply them to missing values:

```
# means dictionary
ageMeans <- ages$Age
names(ageMeans) <- with(ages, paste(Sex, Pclass, Title, sep='-'))

# column of dict keys
combi$Key <- with(combi, paste(Sex, Pclass, Title, sep='-'))

# apply means to missing values using dict keys
combi$AgeNA <- combi$Age
AgeNA <- is.na(combi$AgeNA)
combi$Age[AgeNA] <- sapply(combi$Key[AgeNA], function (x) ageMeans[x])

# check if no remaining missing values
table(is.na(combi$Age))

FALSE 
 1309 
```

<br>

Our dataset is now fully prepared; we can start applying different models to get the best possible predictions.




# PREDICTIONS

Now that our dataset is ready, we can apply a Machine Learning algorithm to make predictions.

But first, we have to split it back:

```
train <- combi[!is.na(combi$Survived),]
test <- combi[is.na(combi$Survived),]
```

One of the most popular Machine Learnign algorithms is Random Forest. Let's train our model and see how well it performs:

```
library(randomForest)

# setting a seed is very important for reproducibility
set.seed(415)

# training
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Title + GroupSize + Fare + Age + Embarked, data=train, importance=TRUE, ntree=2000);fit

# predictions
Prediction <- predict(fit, test, OOB=TRUE, type = "response")

# saving submission
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "../submissions/randForest.csv", row.names = FALSE)
```

The result is not great: it gets us roughly to the third 33% of all submissions.

![rf.png](https://sebastienplat.s3.amazonaws.com/86453403c5116b15b517760c3907cfe71478172965957)

Let's try again with Conditional Forests. They make their decisions in slightly different ways, using a statistical test rather than a purity measure (more information [here](http://stats.stackexchange.com/questions/12140/conditional-inference-trees-vs-traditional-decision-trees)).



```
library(party)

# setting a seed is very important for reproducibility
set.seed(415)

# training
ffit <- cforest(as.factor(Survived) ~ Pclass + Sex + Title + GroupSize + Fare + Age + Embarked, data=train, controls=cforest_unbiased(ntree=2000, mtry=3));fit

# predictions
Prediction <- predict(fit, test, OOB=TRUE, type = "response")

# saving submission
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "../submissions/condForest.csv", row.names = FALSE)
```

The prediction is much better: it brings us to the top 10% !

![cf.png](https://sebastienplat.s3.amazonaws.com/9484c73d86dad85a7cd6f9113495fabd1478173075497)



As a final note, we see that improving accuracy, even slightly, can vastly improve our ranking. However, his particular dataset is very small: correctly predicting one result  has a huge impact on scoring, even if it is not statistically significant.




# APPENDIX

#### Deck Feature

_Note: this feature is in appendix because it is complex to build but do not improve the model accuracy. I think it is still interesting as an illustration of what can be done._

<br>

As seen before, we don't have a lot of information about the cabins. [The Encyclopedia Titanica](https://www.encyclopedia-titanica.org/cabins.html) explains that:

> The allocation of cabins on the Titanic is a source of continuing interest and endless speculation. Apart from the recollections of survivors and a few tickets and boarding cards, the only authoritative source of cabin data is the incomplete first class passenger list recovered with the body of steward Herbert Cave. 

We will assume that all members of a group _(as identified in the previous section)_ had cabins in the same deck.

Some passengers have more than one cabin. Let's have a closer look:

```r
# check cabin values with a space
> cabins <- levels(combi$Cabin)
> cabins[grepl(" ",cabins)]

[1] "B51 B53 B55"     "B57 B59 B63 B66" "B58 B60"         "B82 B84"        
 [5] "B96 B98"         "C22 C26"         "C23 C25 C27"     "C62 C64"        
 [9] "D10 D12"         "F E69"           "F G63"           "F G73"          
[13] "B52 B54 B56"     "C55 C57"         "E39 E41"         "F E46"          
[17] "F E57"   
```

In almost all cases, all listed cabins share the same deck. Only values starting with `F ` are inconsistent, so we will discard `F ` and keep what is left. For example, `F E69` will become `E69`. 

```r
# discard 'F '
combi$Deck <- gsub("F ", "", combi$Cabin)

# keep only the deck letter
combi$Deck <- as.factor(substr(combi$Deck, 1, 1))
```

The table below shows the deck split for each class. As mentionned above, almost no information exists for Second- and Third-Class.

```r
# Deck split for each class 
> table(combi$Pclass,combi$Deck)
   
          A   B   C   D   E   F   G   T
  1  67  22  65  94  40  34   0   0   1
  2 254   0   0   0   6   4  13   0   0
  3 693   0   0   0   0   6   1   9   0
```

_Note: [The Encyclopedia Titanica](https://www.encyclopedia-titanica.org/cabins.html) states that the cabin "T" is the only occupied cabin in the Boat Deck. As it is a separate deck, we will keep the value as is._

Each group travelling together is likely to have accomodations in the same deck. Let's see if we can find missing information using this hypothesis.

```r
# group deck
groupDeck <- data.frame(table(combi$Deck, combi$GroupID))
names(groupDeck) <- c("GroupDeck","GroupID","Freq")

# keep only groups with at least one person on a known deck
groupDeck <- groupDeck[groupDeck$GroupDeck != "" & groupDeck$Freq > 0, ]
groupDeck <- droplevels(groupDeck)
```

We now have a list of all groups with useful deck information.

_Note: Dropping levels removes all groups that did not match our condition of at least one person on a known deck. It takes less memory (which is useful when working with large datasets) and is cleaner when using functions that rely on factor levels, like `table`._


Let's check our hypothesis of one deck per group:

```r
# check if each group is only linked to one deck
> check <- data.frame(table(groupDeck$GroupID))
> dim(check)
[1] 176   2

> check[check$Freq > 1, ]

        Var1 Freq
3     110465    2
786 PC 17485    2
```

Only two groups of two, out of the 176 with useful deck information, have cabins in different decks:

```r
# look at duplicates
combi[combi$GroupID %in% c("110465","PC 17485"), 
      c("GroupID", "GroupSize", "Title", "LastName", 
        "Pclass", "Fare", "Deck","Cabin")]
      
      GroupID GroupSize Title    LastName Pclass    Fare Deck Cabin
7      110465         2    Mr      Porter      1 52.0000    C  C110
8      110465         2    Mr    Clifford      1 52.0000    A   A14
1103 PC 17485         2   Sir Duff Gordon      1 56.9292    A   A20
1104 PC 17485         2  Miss Francatelli      1 56.9292    E   E36
```

Duplicates in `GroupDeck` could cause problems during the merge. Let's remove them:

```r
# remove duplicates
> dim(groupDeck)
[1] 178   3
> groupDeck[groupDeck$GroupID %in% c("110465","PC 17485"), ]

     GroupDeck  GroupID Freq
20           A   110465    1
22           C   110465    1
7067         A PC 17485    1
7071         E PC 17485    1
```

It is strange: even though `groupDeck` only has 178 rows, the extract shows `7067` and `7071`! This is because the row names are not reset by default when subsetting a dataframe. We have to manually reset first:

```r
# reset numerotation
> rownames(groupDeck) <- NULL

# get correct row numbers
> groupDeck[groupDeck$GroupID %in% c("110465","PC 17485"), ]

    GroupDeck  GroupID Freq
3           A   110465    1
4           C   110465    1
143         A PC 17485    1
144         E PC 17485    1

# remove duplicates
> groupDeck <- groupDeck[-c(4,144), ]
```

We can now merge `groupDeck` and `combi`.

```r
# Drop unnecessary Freq column
groupDeck <- groupDeck[, c("GroupID","GroupDeck")]

# merge with combi
combi <- merge(x = combi, y = groupDeck, by = "GroupID", all = TRUE)

# specific case: NA
levels(combi$GroupDeck) <- c(levels(combi$GroupDeck), "") #add level
combi$GroupDeck[is.na(combi$GroupDeck)] <- ""

# Specific case:  tickets "110465" and "PC 17485"
diffDeck <- combi$GroupDeck != "" & combi$Deck != "" &combi$GroupDeck != combi$Deck
combi$GroupDeck[diffDeck] <- combi$Deck[diffDeck]
```

Now that we are done, we can see how much new deck information we have gained:

```r
# check gain
> table(combi$Deck != "",combi$GroupDeck != "")
       
        FALSE TRUE
  FALSE   997   17
  TRUE      0  295
```

All our hard work translated to only 17 new deck values. It is not much, and after testing did not improve our scoring.


