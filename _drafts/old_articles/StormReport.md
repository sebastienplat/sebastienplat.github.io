# OVERVIEW

#### GOAL

The goal of this project it to analyze the **USA Storm Database\***, in order to measure the impact of severe wheather events:

  + identify the **most harmful** types of events:
    + **Fatalities** 
    + **Injuries**
  + identify the types of events that have the **greatest economic consequences**:
    + **Property damage**
    + **Crop damage**
  + present the results in an **interactive dashboard**
 

  <br>
  <br>
  
\* _U.S. National Oceanic and Atmospheric Administration's (NOAA) Storm Database. _

#### DATASET

The raw data is a large database of roughly **one million entries** and 40 variables. The main task was to clean it:
 
   + select the relevant features
   + clean all the selected features (checking outliers, etc.)
   + keep only the relevant entries (at least one harm category > 0)
 

The database used for analysis has only **250k entries** and 15 variables. A detailed explanation of the cleaning method is available in the [In-Depth article](http://www.sebastienplat.com/blog/57c6a66458d0d90300b53e39).

#### RESULTS - HEALTH

+ Tornadoes are the leading cause of Health Issues, but  excessive heat events on average the most dangerous

<img style="margin: 0 auto;width:55%;display:block;" src="https://sebastienplat.s3.amazonaws.com/f23a25ae2f544dc0585e9de8ddb2fc021473958691021" alt="health"></img>

#### RESULTS - DAMAGES

+ Hurricanes are the leading cause of Property Damage, and are on average the most dangerous
  + Drought are the leading cause of Crop damage, but ice storms are on average the most dangerous _(they occur less often, though)_

<img style="margin: 0 auto;width:55%;display:block;" src="https://sebastienplat.s3.amazonaws.com/871e412977b099e3ce15bb9f8a293e1a1473958745141" alt="Dmg"></img>

#### DASHBOARD

The final database of 250k entries can be explored though the interactive dashboard _(desktop view on the left, mobile view on the right)_:

<p>
<img style="width:40%;" src="https://sebastienplat.s3.amazonaws.com/8ce36a645fd8ab26ce0f5ac7e163323f1474292786249" alt="stormReportLG"></img>

<img style="width:25%;" src="https://sebastienplat.s3.amazonaws.com/2ba8d96080a6376c70a3c1fd99e249c51474292797189" alt="stormReportXS"></img>
</p>

The user can filter events by Harm Category _(Fatalities, etc.)_, Event Types _(Thunderstorm, etc.)_ and Year Range. The dashboard is available in the [demo page](http://www.sebastienplat.com/stormReport/demo).





# IN-DEPTH

#### DATASET

**Severe weather events** can cause major public health and economic problems for communities: fatalities, injuries, property and crop damage. Preventing such outcomes is a key concern.

This post will present the analysis of the U.S. National Oceanic and Atmospheric Administration's (NOAA) Storm Database, and its results regarding the following questions:

+ Across the US, which types of events are **most harmful** with respect to population health?
+ Across the US, which types of events have the **greatest economic consequences**?

The answer will show:

+ the **Top10 Event Types** for each Harm Category: 
  + population health: **fatalities and injuries**
  + economic impact: **property damage and crop damage**
+ their **average impact** when harmful

The average impact will facilitate the comparizon between Event Types occuring with very different frequencies.

An interactive map is available in the [demo tab](http://www.sebastienplat.com/stormReport/demo), and a Shiny version is available [here](https://splat.shinyapps.io/stormReport/).

<hr>

_Note: the raw data file can be downloaded [here](https://d396qusza40orc.cloudfront.net/repdata%2Fdata%2FStormData.csv.bz2)._

#### DATA CLEANING

The raw database has more than 900k observations and 37 variables. In this section, we will:

+ select the relevant features
+ clean all the selected features
+ keep only the observations relevant to our analysis

Once done, the final database will have 250k observations and 15 variables only.


#### Load Data

Let's start by downloading the data & reading it:

```r
# load libraries
library(dplyr)
library(ggplot2)

# download archive
if (!dir.exists("data")) {
  dir.create("data")
}

if (!file.exists("data\\stormData.bz2")) {
  download.file("https://d396qusza40orc.cloudfront.net/repdata%2Fdata%2FStormData.csv.bz2",
                "data\\stormData.bz2")
}

# read archive (takes a while)
storm <- read.csv("data\\stormData.bz2")
```

#### Select Variables

We want to keep only the observations and variables that are relevant for this analysis: dates, states, event types and harm categories.

By looking at the documentation ([link](https://d396qusza40orc.cloudfront.net/repdata%2Fpeer2_doc%2Fpd01016005curr.pdf)), and especially section 2.7 p12, we can see there are two types of damage: property & crop. They are expressed with two variables:

+ xxxDMG gives the amount
+ xxxEXP gives the unit: thousands (K), millions (M) or billions (B)

This leads us to select only the following variables:

```r
# creating a year variable
bgn_date <- as.character(storm$BGN_DATE)
storm$YEAR <- factor(as.POSIXlt(strptime(bgn_date, "%m/%d/%Y"))$year+1900)

# selecting fields relevant to the analysis
stormSelect <- select (storm, 
  BGN_DATE, STATE, EVTYPE, 
  FATALITIES, INJURIES, PROPDMG, PROPDMGEXP, CROPDMG, CROPDMGEXP, 
  YEAR
)
```

#### Clean Damage Costs

The following tables show that **the damage unit is not always properly mentioned**: we expect only K, M and B (see above).

```r
# units for property damages > 0 - expecting only K,M,B
t <- table (stormSelect[stormSelect$PROPDMG > 0, c("PROPDMGEXP")])
pandoc.table(t, 
  split.table = Inf,
  emphasize.strong.cols = which(toupper(names(t)) %in% c("B","K","M")),
  style = 'rmarkdown'
)
```

|  &nbsp;  |  -  |  ?  |  +  |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |   B    |  h  |  H  |     K      |   m   |     M     |
|:--------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:------:|:---:|:---:|:----------:|:-----:|:---------:|
| isBordered |
|    76    |  1  |  0  |  5  | 209 |  0  |  1  |  1  |  4  | 18  |  3  |  2  |  0  | **40** |  1  |  6  | **227 481** | **7** | **11 319** |


```r
# units for crop damages > 0 - expecting only K,M,B
t <- table (stormSelect[stormSelect$CROPDMG > 0, c("CROPDMGEXP")])
pandoc.table(t, 
  split.table = Inf,
  emphasize.strong.cols = which(toupper(names(t)) %in% c("B","K","M")),
  style = 'rmarkdown'
)
```

|  &nbsp;  |  ?  |  0  |  2  |   B   |   k    |     K     |   m   |    M     |
|:--------:|:---:|:---:|:---:|:-----:|:------:|:---------:|:-----:|:--------:|
| isBordered |
|    3     |  0  | 12  |  0  | **7** | **21** | **20 137** | **1** | **1 918** |

To adress this issue, we will:

+ consider **entries with incorrect units as having zero damage**
+ create new variables **prop_dmg** and **crop_dmg** that convert damages in USD

```r
# cleaning up property damages: we count them only if they have a proper unit
stormSelect <- mutate (stormSelect, prop_dmg = 
  ifelse(toupper(PROPDMGEXP) == "B", PROPDMG*10^9, 
  ifelse(toupper(PROPDMGEXP) == "M", PROPDMG*10^6,
  ifelse(toupper(PROPDMGEXP) == "K", PROPDMG*10^3, 
  0
))))

# cleaning up crop damages: we count them only if they have a proper unit
stormSelect <- mutate (stormSelect, crop_dmg = 
  ifelse(toupper(CROPDMGEXP) == "B", CROPDMG*10^9, 
  ifelse(toupper(CROPDMGEXP) == "M", CROPDMG*10^6,
  ifelse(toupper(CROPDMGEXP) == "K", CROPDMG*10^3,
  0
))))
```

#### Filter Observations

We will keep only the observations with either casualties/injuries or damage:

```r
# drop all unused observations
stormSelectFilter <- filter (stormSelect, FATALITIES > 0 | 
                                          INJURIES > 0 |
                                          prop_dmg > 0 | 
                                          crop_dmg > 0) %>% 
                     select (BGN_DATE, STATE, EVTYPE, 
                             FATALITIES, INJURIES,
                             PROPDMG, PROPDMGEXP,  prop_dmg, 
                             CROPDMG, CROPDMGEXP, crop_dmg, 
                             year)

# drop all unused levels
stormSelectFilter <- droplevels (stormSelectFilter)

# convert all EVTYPE levels in uppercase
levels(stormSelectFilter$EVTYPE) <- toupper(levels(stormSelectFilter$EVTYPE))
```

Our data frame is now much smaller, with only 250k observations and 12 variables.

#### Cleaning outliers

Fig. 1 shows the fatalities, injuries and damage reported for each recorded event:

```r
# same plot structure for all harm categories
stormPlot <- function (field, threshold, yLabel = field, maxDmg = NULL, dmgCoeff = NULL) {
  p1 = qplot(
    data=stormSelectFilter, 
    x=year, 
    y = stormSelectFilter[, c(field)]
  ) 
  p1 = p1 + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  p1 = p1 + scale_x_discrete(breaks = seq(1950, 2010, 10))
  p1 = p1 + ylab(yLabel)
  p1 = p1 + geom_hline(aes(yintercept=threshold), color="red")
  if (!is.null(maxDmg)) {
    p1 = p1 + scale_y_continuous(
      breaks = seq(0, maxDmg, dmgCoeff*10^9), 
      labels = seq(0, maxDmg/10^9, dmgCoeff)
    )
  }
  return(p1)
}

p1 <- stormPlot("FATALITIES", 200)
p2 <- stormPlot("INJURIES", 1000)
p3 <- stormPlot("prop_dmg", 10*10^9, "Property Damages (in Billion USD)", 1.2*10^11, 10)
p4 <- stormPlot("crop_dmg", 10^9, "Crop Damages (in Billion USD)", 5*10^9, 1)

multiplot(p1,p2,p3,p4,cols=4, title = "Fig. 1: Fatalities & Damages per recorded event")
```

![outliers.png](https://sebastienplat.s3.amazonaws.com/efe7617f45d530abfd948a1dfffcf6e51473870449216)

We clearly see some outliers (points above the red lines). Let's investigate them.


##### Fatalities

```r
# fatalities
pandoc.table(stormSelectFilter[which(stormSelectFilter$FATALITIES > 200), 
                               c("BGN_DATE", "STATE", "EVTYPE", "FATALITIES")], 
             split.table = Inf,
             style = 'rmarkdown')
```

|   &nbsp;    |     BGN_DATE      |  STATE  |  EVTYPE  |  FATALITIES  |
|:-----------:|:-----------------:|:-------:|:--------:|-------------:|
|  **31915**  | 7/12/1995 0:00:00 |   IL    |   HEAT   |     583      |


The deadliest recorded event is the **July 1995 Heat Wave in Chicago** ([link](https://en.wikipedia.org/wiki/1995_Chicago_heat_wave)).


##### Injuries

```r
# Injuries
pandoc.table(stormSelectFilter[which(stormSelectFilter$INJURIES > 1000), 
                               c("BGN_DATE", "STATE", "EVTYPE", "INJURIES")], 
             split.table = Inf,
             style = 'rmarkdown')
```

|    &nbsp;    |     BGN_DATE      |  STATE  |  EVTYPE   |  INJURIES  |
|:------------:|:-----------------:|:-------:|:---------:|-----------:|
|  **11611**   | 6/9/1953 0:00:00  |   MA    |  TORNADO  |    1 228    |
|  **18631**   | 4/3/1974 0:00:00  |   OH    |  TORNADO  |    1 150    |
|  **24585**   | 4/10/1979 0:00:00 |   TX    |  TORNADO  |    1 700    |
|  **43962**   | 2/8/1994 0:00:00  |   OH    | ICE STORM |    1 568    |
|  **241460**  | 5/22/2011 0:00:00 |   MO    |  TORNADO  |    1 150    |


The recorded events with the most injuries are:

+ the **1953 Flint-Worcester tornado outbreak sequence** ([link](https://en.wikipedia.org/wiki/Flint%E2%80%93Worcester_tornado_outbreak_sequence))
+ the **1974 Super Outbreak** ([link](https://en.wikipedia.org/wiki/1974_Super_Outbreak))
+ the **1979 Red River Valley tornado outbreak** ([link](https://en.wikipedia.org/wiki/1979_Red_River_Valley_tornado_outbreak))
+ the **1994 Artic Outbreak** ([link](https://shaneholinde.wordpress.com/2014/01/07/remembering-januaryfebruary-1994-winters-tko/))
+ the **2011 tornado outbreak** ([link](https://en.wikipedia.org/wiki/May_21%E2%80%9326,_2011_tornado_outbreak_sequence))


##### Property Damage

```r
# property damage
pandoc.table(stormSelectFilter[which(stormSelectFilter$prop_dmg >= 10*10^9), 
                               c("BGN_DATE", "STATE", "EVTYPE", "PROPDMG", "PROPDMGEXP")], 
             split.table = Inf,
             style = 'rmarkdown')
```

|    &nbsp;    |      BGN_DATE      |  STATE  |      EVTYPE       |  PROPDMG  |  PROPDMGEXP  |
|:------------:|:------------------:|:-------:|:-----------------:|----------:|:-------------:|
|  **153302**  | 10/24/2005 0:00:00 |   FL    | HURRICANE/TYPHOON |    10     |      B       |
|  **155490**  | 8/28/2005 0:00:00  |   LA    | HURRICANE/TYPHOON |   16.93   |      B       |
|  **155491**  | 8/29/2005 0:00:00  |   LA    |    STORM SURGE    |   31.3    |      B       |
|  **156656**  | 8/29/2005 0:00:00  |   MS    |    STORM SURGE    |   11.26   |      B       |
|  **162532**  |  1/1/2006 0:00:00  |   CA    |       FLOOD       |    115    |      B       |


The recorded events with the most property damage are:

+ **Hurricane Wilma in 2005 in Florida** ([link](https://en.wikipedia.org/wiki/Hurricane_Wilma)), the most intense tropical cyclone ever recorded in the Atlantic basin
+ **Hurricane Katrina in 2005** ([link](https://en.wikipedia.org/wiki/Hurricane_Katrina)), identified as the costliest natural disaster in the History of the USA
+ **the flood of january 2006 in California**

**The reported damage of the California flood will be considered a typo**, as it is estimated by other sources as approx. 300 Millions USD ([link](http://pubs.usgs.gov/of/2006/1182/)), and because it seems unreasonable to assume it cost 10 times more than Katrina.

We will convert the damage as being **115 millions USD** instead:

```r
# converting billions to millions for the CA flood of 2006
stormSelectFilter[which(stormSelectFilter$prop_dmg > 10^11),]$PROPDMGEXP <- "M"
stormSelectFilter[which(stormSelectFilter$prop_dmg > 10^11),]$prop_dmg <- 115*10^6
```


##### Crop Damage

```r
# crop damage
pandoc.table(stormSelectFilter[which(stormSelectFilter$crop_dmg >= 10^9), 
                               c("BGN_DATE", "STATE", "EVTYPE", "CROPDMG", "CROPDMGEXP")], 
             split.table = Inf,
             style = 'rmarkdown')
```

|    &nbsp;    |     BGN_DATE      |  STATE  |      EVTYPE       |  CROPDMG  |  CROPDMGEXP  |
|:------------:|:-----------------:|:-------:|:-----------------:|----------:|:-------------:|
|  **31770**   | 8/31/1993 0:00:00 |   IL    |    RIVER FLOOD    |     5     |      B       |
|  **37985**   | 2/9/1994 0:00:00  |   MS    |     ICE STORM     |     5     |      B       |
|  **156657**  | 8/29/2005 0:00:00 |   MS    | HURRICANE/TYPHOON |   1.51    |      B       |
|  **171518**  | 1/1/2006 0:00:00  |   TX    |      DROUGHT      |     1     |      B       |


The recorded events with the most crop damage are:

+ the **Great Flood of 1993** ([link](https://en.wikipedia.org/wiki/Great_Flood_of_1993))
+ the **Southeast Ice Storm of 1994** ([link](http://www.alabamawx.com/?p=5469))
+ **Hurricane Katrina in 2005**
+ a **severe drought in Texas in 2005-2006** ([link](http://twri.tamu.edu/publications/txh2o/fall-2011/timeline-of-droughts-in-texas/))


### Cleaning Event Types

The documentation stipulates that there are **48 Event Types** one can use to describe an event (section 7 - Event Types, p18). The events are largely reported in this normalized fashion, but there are also many cases where the guidelines are not fully respected.

In an attempt to facilitate the reading of the results, we will **map all the events to the 48 Event Types** by using keywords and the following rules:

+ a vector of **values to replace** is created for each of the 48 Event Types
+ the **more precise classification** is listed **first** ("marine thunderstorm" before "thunderstorm", etc.)
+ the **deadliest / costliest events** are listed **first**, to catch events that could be affected to different Event Types ("Avalanche + Blizzard" for example)
+ the few events without a clear Event Type will be **classified as "OTHER"**

The first few examples are listed below:

```r
# EVTYPE list: EVTYPE levels (only 430 rows, much faster than working with the full table)
EVTYPE_list <- data.frame(EVTYPE=levels(stormSelectFilter$EVTYPE))

# normalizing event types using the Storm Data Event Table
EVTYPE_list$Event_Type <- decodeList (
    EVTYPE_list$EVTYPE, 
    "OTHER", 
    list(
        c("ASTRONOMICAL LOW TIDE"), "ASTRONOMICAL LOW TIDE",
        c("AVALANCHE", "AVALANCE"), "AVALANCHE",
        #[...]
    )
)

# merging the Event Type list with the main dataset
stormSelectFilterGroup <- left_join(stormSelectFilter, EVTYPE_list, by="EVTYPE")

# converting values into factors
stormSelectFilterGroup$Event_Type <- factor(stormSelectFilterGroup$Event_Type)
```

_Note: The details of the "decodeList" function are in the Appendix tab._



# ANALYSIS

Now that our Data Processing is complete, we can focus on our two questions:

+ Across the US, which types of events are most harmful with respect to population health?
+ Across the US, which types of events have the greatest economic consequences?

We will focus on the Top 10 Event Types for each Harm Category, to facilitate the analysis.

#### Aggregation Function

```r
aggregateHarmType <- function (harmType, strongLimit) {
  
  # aggregate by event type
  # we use the SE version of summarize to pass it a variable
  df <- as.data.frame(
    stormSelectFilterGroup %>%
    group_by(Event_Type) %>%
    summarize_(
      Count = paste0("length(which(", harmType, " > 0))"),
      Total = paste0("sum(", harmType, ")")
    )
  )
  
  # identify top 10
  df <- arrange(df,desc(Total))
  df$Top10 <- ifelse(row(df)<10, as.character(df$Event_Type), "OTHER")[,1]
  
  # group the other event types as "OTHER"
  df <- as.data.frame(
    df %>%
    group_by(Top10) %>% 
    summarize(
      Count = sum(Count),
      Total = sum(Total)
    )
  ) %>%
  arrange(desc(Total))
  
  # estimate avg by harmful event
  df$Average <- round(with(df, Total/Count),1)
  
  # move "OTHER" at the end
  df <- rbind (
    df[-which(df$Top10=="OTHER"), ],
    df[ which(df$Top10=="OTHER"), ]
  )
  rownames(df) <- NULL
  
  # sort factors by decr value
  df$Top10 <- with(df, factor(Top10, levels=Top10[1:10]))
  
  # convert damage in B and M
  if (harmType == "prop_dmg" || harmType == "crop_dmg") {
    df$PrettyTotal   <- paste(format(round(df$Total/10^9,1),   nsmall=1), "B")
    df$PrettyAverage <- paste(format(round(df$Average/10^6,1), nsmall=1), "M")
  }
  else {
    df$PrettyTotal   <- df$Total
    df$PrettyAverage <- df$Average
  }
  
  # show synthesis
  pandoc.table(
    select(df, -Total, -Average), 
    split.table = Inf, justify="right", 
    emphasize.strong.rows = which(df$Average > strongLimit,  arr.ind = TRUE),
    style = 'rmarkdown'
  )
  
  # plot: points of diff sizes
  p1 = ggplot(df, aes(x = Top10, y = Total)) 
  p1 = p1 + geom_point(aes(colour = Average, size = Average))
  
  # merges legends & move titles to the top
  p1 = p1 + guides(
    colour = guide_legend(title.position = "top"),
    size =   guide_legend(title.position = "top")
  )
  
  # format axis & legend numbers
  prettify <- function(number, coeff, letter) {
    return(paste(format(round(number/coeff, 1), trim = TRUE), letter))
  }
  
  prettifyNumbers <- function(list) {
    list <- ifelse(list >= 10^9, prettify(list, 1e9, "B"),
            ifelse(list >= 10^6, prettify(list, 1e6, "M"),
            ifelse(list >= 10^3, prettify(list, 1e3, "k"),
            list)))
  }
  
  # range increases the size of dots for easing reading
  p1 = p1 + scale_y_continuous(labels = prettifyNumbers)
  p1 = p1 + scale_colour_continuous(labels = prettifyNumbers)
  p1 = p1 + scale_size_continuous(labels = prettifyNumbers, range=c(1,10))
  
  # forcefully include 0 in the y axis
  p1 = p1 + expand_limits(y=0)
  
  # general theming
  p1 = p1 + theme(
    plot.title =  element_text(face='italic', vjust=2),
    plot.margin = unit(c(1,1,1,1), "cm"),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
    legend.position="bottom",
    text = element_text(size=16)
  )
  
  # plot titles
  p1 = p1 + ggtitle (harmType) 
  p1 = p1  + xlab("") + ylab("")

  return(p1)
  
}

p1 <- aggregateHarmType ("FATALITIES", 3)
p2 <- aggregateHarmType ("INJURIES", 30)
p3 <- aggregateHarmType ("prop_dmg", 200e6)
p4 <- aggregateHarmType ("crop_dmg", 50e6)

multiplot (p1,p2, cols=2, title="Fig. 2: Fatalities & Injuries per Event Type, 1950-2011")
multiplot (p3,p4, cols=2, title="Fig. 3: Damage per Event Type, 1950-2011")
```

#### Results

##### Tables

 Are included in the tables below:

+ the **count** of events where the Harm Category occured (for example, only a small percentage of events have led to fatalities)
+ the **total** for the period 1950-2011
+ the **average** per event

<div>
<uib-tabset active="activeJustified" justified="true">
    <uib-tab index="0" heading="Fatalities">
    	!!
		|                   Top10 |    Count |    Total |   Average |
		|------------------------:|---------:|---------:|----------:|
		|             **TORNADO** | **1 604** | **5 661** |   **3.5** |
		|      **EXCESSIVE HEAT** |  **591** | **2 058** |   **3.5** |
		|                **HEAT** |  **205** | **1 114** |   **5.4** |
		|             FLASH FLOOD |      671 |     1 035 |       1.5 |
		|               LIGHTNING |      760 |      817 |       1.1 |
		|       THUNDERSTORM WIND |      585 |      721 |       1.2 |
		|             RIP CURRENT |      508 |      572 |       1.1 |
		|                   FLOOD |      333 |      512 |       1.5 |
		| EXTREME COLD/WIND CHILL |      212 |      307 |       1.4 |
		|                   OTHER |     1 505 |     2 348 |       1.6 |
		!!
    </uib-tab>
    <uib-tab index="1" heading="Injuries">
    	!!
        |              Top10 |   Count |    Total |   Average |
        |-------------------:|--------:|---------:|----------:|
        |            TORNADO |    7 710 |    91 407 |      11.9 |
        |  THUNDERSTORM WIND |    3 655 |     9 536 |       2.6 |
        |          **FLOOD** | **187** | **6 874** |  **36.8** |
        | **EXCESSIVE HEAT** | **175** | **6 747** |  **38.6** |
        |          LIGHTNING |    2 810 |     5 231 |       1.9 |
        |           **HEAT** |  **56** | **2 479** |  **44.3** |
        |      **ICE STORM** |  **64** | **1 992** |  **31.1** |
        |        FLASH FLOOD |     393 |     1 800 |       4.6 |
        |           WILDFIRE |     315 |     1 606 |       5.1 |
        |              OTHER |    2 239 |    12 856 |       5.7 |
		!!
    </uib-tab>
    <uib-tab index="2" heading="Property Damage">
    	!!
        |                 Top10 |   Count |      Total |     Average |
        |----------------------:|--------:|-----------:|------------:|
        | **HURRICANE/TYPHOON** | **218** | **85.4 B** | **391.5 M** |
        |               TORNADO |   39 049 |     58.6 B |       1.5 M |
        |  **STORM SURGE/TIDE** | **220** | **48.0 B** | **218.0 M** |
        |                 FLOOD |   10 821 |     35.4 B |       3.3 M |
        |           FLASH FLOOD |   21 145 |     16.9 B |       0.8 M |
        |                  HAIL |   23 046 |     16.0 B |       0.7 M |
        |     THUNDERSTORM WIND |  116 089 |     11.0 B |       0.1 M |
        |              WILDFIRE |    1 052 |      8.5 B |       8.1 M |
        |        TROPICAL STORM |     401 |      7.7 B |      19.2 M |
        |                 OTHER |   26 806 |     25.1 B |       0.9 M |
		!!
    </uib-tab>
    <uib-tab index="3" heading="Crop Damage">
    	!!
        |                   Top10 |   Count |      Total |     Average |
        |------------------------:|--------:|-----------:|------------:|
        |             **DROUGHT** | **261** | **14.0 B** |  **53.5 M** |
        |                   FLOOD |    2 041 |     10.9 B |       5.3 M |
        |   **HURRICANE/TYPHOON** |  **95** |  **5.5 B** |  **58.1 M** |
        |           **ICE STORM** |  **24** |  **5.0 B** | **209.3 M** |
        |                    HAIL |    9 391 |      3.0 B |       0.3 M |
        |            FROST/FREEZE |     137 |      2.0 B |      14.6 M |
        |             FLASH FLOOD |    2 209 |      1.5 B |       0.7 M |
        | EXTREME COLD/WIND CHILL |      57 |      1.4 B |      23.9 M |
        |       THUNDERSTORM WIND |    5 437 |      1.3 B |       0.2 M |
        |                   OTHER |    2 432 |      4.5 B |       1.9 M |
		!!
    </uib-tab>
  </uib-tabset>
</div>


These tables show that:

+ **Fatalities & Injuries**
    + Tornadoes are the leading cause
    + But on average, excessive heat is more dangerous
+ **Property damage**
    + Hurricanes are the leading cause
    + And on average, they are the most dangerous
+ **Crop damage**
    + Drought are the leading cause
    + But on average, ice storms are the most dangerous

The discrepancies between total and average are explained by the fact that some events are much more likely to accur than others.

##### Graph Synthesis

The graphs produced by the aggregate function sum up our conclusions nicely:

![FatInj.png](https://sebastienplat.s3.amazonaws.com/f23a25ae2f544dc0585e9de8ddb2fc021473958691021)

![Dmg.png](https://sebastienplat.s3.amazonaws.com/871e412977b099e3ce15bb9f8a293e1a1473958745141)


#### Final comments

+ Averages could have been calculated as total / number of events, but it is not clear what triggers a report for events without damage nor casualties. It seems that thunderstorms are reported much more frequently than other events (maybe because they are easier to spot ?) so such a ratio could induce a bias.
+ Some event types in "OTHER" may have an average higher than the top10, but overall they have a much smaller impact. So it seemed relevant no to highlight them.
 