# INTRODUCTION

This project aims to create a program that makes it easier for people to type on their mobile devices, by suggesting three options for what the next word might be. It should approximate the smart keyboard made by SwiftKey, our corporate partner in this capstone. 

It will be based on predictive models, built from large amounts of data collected on the internet (the corpus is called [HC Corpora](http://www.corpora.heliohost.org/)):

+ newspapers and magazines
+ personal and professional blogs
+ Twitter updates

When building our models, we will focus on:

+ **speed**: predictions should not take more than one second to appear
+ **accuracy**: predictions should be relevant to the user
+ **correctness**: predictions should be actual english words

The last point is not to be neglected, especially for data coming from Twitter (use of hashtags, spelling errors, etc.).




# CLEANING DATA

#### Sampling

The [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus), which has been created in the 1960's and which for many years has been among the most-cited resources in the field, contains only one million words. Contemporary corpora tend to be much larger, on the order of 100 million words.

Our corpus totals more than **100 million words**. Working with such a large amount of data can be computationally intensive, so we will keep only a **10% sample**, or approx. **10 million words**. Our three sources are roughly the same size. As we want each to have the same weight in our training set, we will keep:

+ 10% of the newspapers and magazines data
+ 10% of the personal and professional blogs data
+ 10% of Twitter updates

This will give us roughly **3 million words per source**.

In order to limit the bias during sampling, we will:

1. fully randomize the lines of each sampled text
2. give each line 10% chance to be selected
3. save our sample in a new file

*Note: we will set seeds for each step, which will guarantee the reproducibility of the sampling.*

<hr>

The files were encoded in [UTF-8](https://en.wikipedia.org/wiki/UTF-8). The sentence tokenizer does not handle non-ASCII characters very well, but we can manually convert the most frequent ones using this [table](http://www.i18nqa.com/debug/utf8-debug.html) and the `gsub`function:

```r
# read the source file
con <- file(src, "rb")
sampleText <- readLines(con, encoding = "ISO-8859-1")
close(con)

# convert UTF-8 characters to ASCII ones
sampleText <- gsub("\u00E2\u20AC\u00A6", '...', sampleText)
sampleText <- gsub("\u00E2\u20AC\u02DC", "'", sampleText)
sampleText <- gsub("\u00E2\u20AC\u2122", "'", sampleText)
sampleText <- gsub("\u00E2\u20AC\u201C", '-', sampleText)
sampleText <- gsub("\u00E2\u20AC\u201D", '-', sampleText)
sampleText <- gsub("\u00E2\u20AC\u0153", '"', sampleText)
sampleText <- gsub("\u00E2\u20AC\u009d", '"', sampleText)
```

#### Sentence tokenization

Once the sample is built, we use a sentence tokenizer to have one sentence per line. This will prevent mixing end & beginning of sentences to form n-grams during the next step.

For example, let's consider an actual extract from our corpus:

> The area has apparently seen better days. The streets are broad, beautiful and tree-lined. The rows of terraces are attractive, though most of the fresh paint is graffiti.

If we did not use a sentence tokenizer, we would have the following four-grams:

  + better days the streets
  + and tree-lined the rows
  
These nonsense n-grams would degrade the quality of our model.


```r
library(NLP)
library(openNLP)
sent_token_annotator <- Maxent_Sent_Token_Annotator()

fullText <- character(length = 0)

con <- file(src, "rb")
while(TRUE) {
    
    # prepare chunk for tokenization
    chunk <- tolower(readLines(con, encoding = "ISO-8859-1", n=1000))
    if ( length(chunk) == 0 ) break
    chunk <- as.String(chunk)
    
    # annotate
    a1 <- annotate(chunk, sent_token_annotator)
    chunk <- chunk[a1]
    
    # split chunk's sentences with ellipsis
    token <- gsub("(\\.){3}",".\n",chunk)
    
    # append split chunk to full text
    fullText <- c(fullText, token)
        
}
close(con)

```




# BUILDING NGRAM TABLES

Now that we have one sentence per line, we can build our model.

#### Tokenization

It will be based on N-gram probabilities, so our first step is to tokenize our samples and **build frequency tables for uni- to 4-grams**. We use the [RWeka package](https://cran.r-project.org/web/packages/RWeka/RWeka.pdf) , as it is up to 100 times faster than others. It splits words like `don't` in two tokens `don`and `t`by default, so we use a custom parsing  method where we remove `'` from the delimiters and add a few more. 

#### N-Gram Lists

Here is our training set after the cleanup:

| N-gram | Unique Before | 
|:-------|--------------:|
| unigrams |     52 031  | 
| bigrams  |  2 001 316  |  
| trigrams |  4 488 650  | 
| 4-grams  |  5 598 643  | 


Here is the list of top10 N-grams for the cleaned-up training set (excl. &lt;UNK&gt;):

| Top10 | 1-gram | 2-gram   | 3-gram | 4-gram |
|:-----:|:-------|:---------|:-------|:-------|
|     1 |    the |   of the |      one of the |         the end of the |
|     2 |     to |   in the |        a lot of |          at the end of |
|     3 |    and |   to the |  thanks for the |  thanks for the follow |
|     4 |      a |  for the |     going to be |        the rest of the |
|     5 |     of |   on the |         to be a |     for the first time |
|     6 |      i |    to be |       i want to |       at the same time |
|     7 |     in |   at the |      the end of |         is going to be |
|     8 |    for |  and the |      out of the |          is one of the |
|     9 |     is |     in a |        it was a |        one of the most |
|    10 |   that | with the |      as well as |       in the middle of |


As an example of 3-gram, and in anticipation to the next steps, here is the 3 most frequent words occuring in the training set after "on my...":

| Word | Count | 
|:-------|-----:|
| ...way  |  227 | 
| ...mind |   92 |  
| ...own  |   91 |




# PERFORMANCE

#### Accuracy Measurement

A smart keyboard is useful to the user when she barely has to type at all. So we will measure the accuracy of our model by calculating how many letters our suggestions save.

For instance, let's say our user wants to type the word `hungry`. This exact word does not appear in the suggestions at first, not until she types `hu`. She has to type only two letters out of six: she saves four letters, 67% of the total.

For the accurary test, we sill use a **18k words sample**, randomly drawn from our data set.


#### Optimization

The N-gram lists can quickly become very large, with a lot of n-grams occurring only once in the training set:

We can reduce their size by filtering n-grams with rare occurrences. We will experiment several thresholds, to find a good compromise between accuracy and speed:

+ **one threshold for unigrams**: we do not want our model to ignore too many words 
+ **one threshold for the other n-grams**

All n-grams occurring with a frequency equal or below the threshold are excluded from our model.


#### Comparizon

We compare three configurations with different thresholds. We will use the notation `xx/yy`: threshold of xx for unigrams and yy for the other n-grams.

+ **0/1**: we keep all unigrams, plus all other n-grams occuring at least once
+ **1/10**: we keep all unigrams occuring at least once, plus  all other n-grams occuring at least ten times
+ **10/10**: we keep all n-grams occuring at least ten times

The results are:

<h5 class="text-center">Number of words per required user inputs</h5>
<div google-chart chart='{
options: {
  title: "Number of words per required inputs",
  titlePosition: "none",
  legend: "none",
  colors:["#214478", "rgb(51, 102, 204)", "rgb(176, 196, 226)"],
  chartArea:{top:30}
},
type: "ColumnChart",
data: {
					"cols": [
						{id: "t", label: "Typed Letters", type: "string"},
						{id: "s", label: "0/1", type: "number"},
                        {id: "s", label: "1/10", type: "number"},
                        {id: "s", label: "10/10", type: "number"}
					], 
					"rows": [
						{c: [
								{v: "0 letter"},
								{v: 4505},
                                {v: 4063},
                                {v: 4063}
						]},
						{c: [
								{v: "1 letter"},
								{v:  4034},
                                {v:  3806},
                                {v:  3806}
						]},
						{c: [
								{v: "2 letters"},
								{v:   3060},
                                {v:  3025},
                                {v:  3029}
						]},
						{c: [
								{v: "3+ letters"},
								{v: 6139},
                                {v: 6840},
                                {v: 6844}
						]}
					]
				}
}' style="height: 300px; width:100%;"></div>

<div class="row">
<div class="col-md-6">
<h5 class="text-center">% of saved letters</h5>
<div google-chart chart='{
options: {
  title: "Saved letters",
  titlePosition: "none",
  legend: "none",
  colors:["#214478", "rgb(51, 102, 204)", "rgb(176, 196, 226)"],
  vAxis: { minValue: 0, maxValue: 1, format: "percent" },
  chartArea:{top:30}
},
type: "ColumnChart",
data: {
					"cols": [
						{id: "t", label: "% of saved letters", type: "string"},
						{id: "s", label: "0/1", type: "number"},
                        {id: "s", label: "1/10", type: "number"},
                        {id: "s", label: "10/10", type: "number"}
					], 
					"rows": [
						{c: [
								{v: "% of saved letters"},
								{v: 1 - 36565/79477},
                                {v: 1 - 39510/79477},
                                {v: 1 - 40282/79477}
						]}
					]
				}
}' style="height: 300px; width:100%;"></div>
</div>
<div class="col-md-6">
<h5 class="text-center">Average time per word (ms)</h5>
<div google-chart chart='{
options: {
  title: "Average time per word (ms)",
  titlePosition: "none",
  legend: "none",
  colors:["#214478", "rgb(51, 102, 204)", "rgb(176, 196, 226)"],
  vAxis: { minValue: 0 },
  chartArea:{top:30}
},
type: "ColumnChart",
data: {
					"cols": [
						{id: "t", label: "Avg Time", type: "string"},
						{id: "s", label: "0/1", type: "number"},
                        {id: "s", label: "1/10", type: "number"},
                        {id: "s", label: "10/10", type: "number"}
					], 
					"rows": [
						{c: [
								{v: "Avg Time (ms)"},
								{v: 1062/17738*1000},
                                {v: 377/17738*1000},
                                {v: 172/17738*1000}
						]}
					]
				}
}' style="height: 300px; width:100%;"></div>
</div>
</div>

We see that all three configurations perform roughly the same, but with were different running times. 

The most complete configuration _(0/1 threshold)_ saves the user **54%** of typing, compared to **49%** for the smallest configuration _(10/10 threshold)_. But it takes **6 times longer** for the model to get these results !

In conclusion, **we will use the 10/10 configuration** for our app: it offers the best compromise between accuracy and speed, and offers the smoothest user experience.




