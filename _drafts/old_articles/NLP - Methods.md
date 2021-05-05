# PROBABILISTIC LANGUAGE MODELS

##### Definition

+ assign a probability to a sentence or a sequence of words
+ related task: probability of an upcoming word

We use the Chain Rule of Probability:

$$P(x_1,...,x_n) = P(x_1)~P(x_2|x_1)~...~P(x_n|x_1,...,x_{n-1}) = \prod_i P(x_i|x_1,x_2,...,x_{i-1}) $$

<br />
Example: P(its water is so transparent) =  

+ P(its) ×  
+ P(water|its) ×  
+ P(is|its water) ×  
+ P(so|its water is) ×  
+ P(transparent|its water is so)

##### Markov assumptions, N-Grams

The number of possibles combinations is too great to be calculated, so we need a simplifying assumption: the probability can be approximated by looking only at the preceding word(s). 

$$P(x_1,...,x_n) = \prod_i P(x_i|x_1,x_2...x_{i-1}) \simeq \prod_i P(x_i|x_{i-k},...,x_{i-1})$$

<br />
We talk about **N-grams models** to describe this simplifying assumption: the number of words used to calculate each probability.

Example: P(that | its water is so transparent) $\simeq$
+ unigram: P(that)
+ bigram: P(that | transparent)  
+ etc.

We can extend to trigrams, 4-grams, etc. But it can become computationally expensive very quickly, and this is an insufficient LM due to long-distance dependencies.

> “The computer which I had just put into the machine room on the fifth floor crashed.”


# N-GRAM PROBABILITIES

##### Maximum Likelihood Estimate

For a bigram:

$$P(w_i|w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1}}$$

It means that we need to count all the unigrams & bigrams of our corpora. Let's say we have $$n$$ unigrams. We can build a $n \times n$ matrix counting all bigrams, then divide by our unigrams count to have the probabilities.

Raw bigram count:

![raw_bigram.png](https://sebastienplat.s3.amazonaws.com/e59714e5889368706e6818ab3a6ca82f1475850595657)

Raw unigram count: 

![raw_unigram.png](https://sebastienplat.s3.amazonaws.com/a0ce525d86a968f13dec8e1fa1b33fb61475850612967)

Bigram probabilities

![bigram_proba.png](https://sebastienplat.s3.amazonaws.com/8b7aba91ba9e746d95ba7f33e060337d1475850624910)

The most likely sentence can be obtained by choosing bigram $(/beg/, w_1)$ with the highest probability, then $(w_1, w_2)$ with the highest probability, and so on until we reach /end/.

Example:

![chain_sentence.png](https://sebastienplat.s3.amazonaws.com/7668b36eaefd9db715e5b70b2358a5311475850636296)
                          
This gives the sentence "I want to eat Chinese food".

Note: it is better to calculate the probability of a sentence in **log space**:

+ it prevends underflows, as probabilities are very small
+ it is faster to compute (additions vs multiplications)


##### Overfitting & Generalization

N-grams are prone to overfitting: they work well for word prediction if the test corpus looks like the training corpus, which often doesn't. This is why we need to build models that generalize for things that never occur in the training set (ie. have a probability of 0), but might in the test set.



# DEALING WITH UNKNOWNS

#### Introduction
There are several methods to account for unknown words/N-grams, ie. words never seen in the training set. They are all based on the same principle: we can free some probability mass by slightly reducing all the calculated probabilities, and use it to account for not-yet encountered N-grams.

Example: $P(w | denied~the)$

| w          | init count | smoothing |
|:-----------|:---:|:---:|
|allegations |3    |2.5  |
|reports     |2    |1.5  |
|claims      |1    |0.5  |
|request     |1    |0.5  |
|**others**  |**0**|**2**|
|TOTAL       |7    |7    |


#### Add-One (Laplace) Smoothing
We saw that to count our bigrams, we built a $n \times n$ matrix that had many zeros, where $n$ is the number of unigrams in our corpus. The simplest smoothing consists of pretending we saw each bigram one more time than we did, which means that:

+ all bigrams $(w_{i-1}, w_i)$ now have a count of at least one
+ all unigrams have $n$ possible bigrams, so their count is increased by $n$    

It gives us a new probability for each bigram:

$$ P^*( w_i | w_{i-1} ) = \frac{ c(w_{i-1}, w_i) + 1}{ c(w_{i-1}) + n} $$

Which in turn gives us reconstituted counts:

$$ c^*( w_{i-1}, w_i ) = c( w_{i-1} ) \times  P^*( w_i | w_{i-1} ) $$

The reconstitued counts are drastically different from the normal counts, because there are many zeros in the matrix. This is why add-1 is not used for N-grams. 

#### Advanced smoothing algorithms
Many smoothing algorithms use the count of things **we've seen once** to estimate the count of things **we've never seen**, as it is likely they will only appear once in the test set.

We will use the frequency of frequencies $N_c$ = the number of N-grams that appear c times in our training set.

##### Good-Turing smoothing
The probability of each new N-gram is the same as the probability of an N-gram that we saw just once:

$$ P^{*}_{GT}(unseen) = \frac{N_1}{N}~~where~N = \sum(N-grams) = \sum(c \times N_c)$$

<br/>
To free the probability mass required for unseen N-grams, all the other probabilities are decreased. The probability of an N-gram seen c times becomes:

$$ P^{*}_{GT}(c) = \frac{c^*}{N}~~where~c^* = \frac{(c+1) \times N_{c+1}}{N_c}$$

<br/>
This can be intuited by a leave-one-out validation (more details [here](https://class.coursera.org/nlp/lecture/32)).

The Good-Turing smoothing show limits for large k:s some frequencies never occur, so in some cases $N_{k+1} = 0$. The Simple Good-Turing [Gale and Sampson] replace empirical $N_k$ with a best-fit power law once counts get unreliable.

##### Absolute Discounting Interpolation


<hr>
#### Vocabulary & OOV words
We can train our model to account for unknown words, or Out Of Vocabulary (OOV) words:

+ Create a fixed lexicon L of size V
+ During tokenization, change to /UNK/ all training words not in L
+ Train /UNK/ probabilities like a normal word
+ At text input, use /UNK/ probabilities for any word not in the training set (ie. not in L)


# IMPROVING THE MODEL

#### Interpolation
We can mix unigram, bigram, trigram to improve our model's efficiency.

##### Simple interpolation
We can use fixed coefficients to weight the use of uni-, bi- and trigrams:

$$ \hat{P}( w_i | w_{i-1}, w_{i-2} ) = \lambda_1 P(w_i | w_{i-1}, w_{i-2}) + \lambda_2 P(w_i | w_{i-1}) + \lambda_3 P(w_i)  $$

##### Conditional interpolation
We can also use coefficients that depend on the previous words:

$$ \hat{P}( w_i | w_{i-1}, w_{i-2} ) = 
\lambda_1 (w^{n-1}_{n-2}) P(w_i | w_{i-1}, w_{i-2}) + 
\lambda_2 (w^{n-1}_{n-2}) P(w_i | w_{i-1}) + 
\lambda_3 (w^{n-1}_{n-2}) P(w_i)  $$

##### Setting lambdas
We split our corpus into three chunks:

+ training set
+ held-out data
+ test set

We choose λs to maximize the probability of held-out data:

+ Fix the N-gram probabilities using the training set
+ Then search for λs that give the largest probability to the held-out set

<hr>

#### Backoff
Use trigram if you have good evidence, otherwise bigram, otherwise unigram. It is usually less efficient than interpolation.

##### Stupid backoff
For large corpora, the stupid backoff gives excellent results: the $S$ value of an n-gram is calculated as follows:

$$ S(w_i | w^{i-1}_{i-n+1}) = $$  
$$\frac{c(w^{i}_{i-n+1})}{c(w^{i-1}_{i-n+1})}~~if~c(^i_{i-n+1}) > 0$$  
$$ 0.4 S(w_i | w^{i-1}_{i-n+2})~~otherwise: (n-1)-gram$$

<br/>
Example with "it is a":

$$ S (a | it~is) =  $$  
$$\frac{c(it~is~a)}{c(it~is)}~~if~c(it~is~a) > 0$$  
$$ 0.4 S(a | is)~~otherwise$$

$$ S (a | is) = $$  
$$\frac{c(is~a)}{c(is)}~~if~c(is~a) > 0$$  
$$0.4 S(a)~~otherwise$$

<br/>

When reaching the unigrams, N being the number of distinct unigams:

$$ S(w_i) = \frac{c(w_i)}{N} $$

<br/>
<hr>
#### Advanced Lanugage Modeling

+ Discriminative models: choose n-gram weights to improve a task, not to fit the training set
+ Parsing-based models
+ Caching models: give an higher weight to recently used words

<hr>

#### Optimization
Dealing with large corpora is very computationally expensive, so some optimization is required.

##### Pruning
+ Only store N-grams with count > threshold
+ Remove singletons of higher-order N-grams
+ Consider Entropy-based pruning: remove N-grams that do not improve the perplexity

##### Efficiency
+ Efficient data structures like tries
+ Bloom filters: approximate language models
+ Store words as indexes, not strings
+ Use Huffman coding to fit large numbers of words into two bytes
+ Quantize probabilities (4-8 bits instead of 8-byte float)


# EVALUATION & PERPLEXITY

##### Extrinsic vs Intrinsic evaluation

**Intrinsic evaluation** considers an isolated NLP system and characterizes its performance with respect to a *gold standard* result as defined by the evaluators. 
**Extrinsic evaluation**, also called *evaluation in use,* considers the NLP system in a more complex setting as either an embedded system or a precise function for a human user. The extrinsic performance of the system is then characterized in terms of utility with respect to the overall task of the extraneous system or the human user. 

For example, consider a syntactic parser which is based on the output of some part of speech (POS) tagger. An intrinsic evaluation would run the POS tagger on [structured data](https://en.wikipedia.org/wiki/Data_model), and compare the system output of the POS tagger to the gold standard output. An extrinsic evaluation would run the parser with some other POS tagger, and then with the novel POS tagger, and compare the parsing accuracy.


##### Perplexity

Perplexity is usually a bad approximation of the model's performance, unless the test data looks just like the training data. So it is generally only useful in pilot experiments

Perplexity $PP$ is the probability of the test set $W$, normalized by the number of words:

$$PP(W) = P(w_1,w_2,...,w_N)^{-1/N} =  \sqrt[N]{\frac{1}{\prod_i P(w_i|w_1,w_2,...,w_{i-1})}} $$

So for bigrams:

$$PP(W) = \sqrt[N]{\frac{1}{\prod_i P(w_i|w_{i-1})}} $$

Note: Minimizing perplexity is the same as maximizing probability.


