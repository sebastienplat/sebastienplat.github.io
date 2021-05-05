# INTRODUCTION

The goal of this project it to create a **smart keyboard** similar to switfkey. 

It should be:

+ **fast**: the suggested words should appear as quickly as the user types
+ **accurate**: the user should only have to type a few letters per word - if at all
+ **mobile-first**: the app should work flawlessly in all screen sizes


I use a predictive model based on **n-gram frequencies** & **stupid backoff**, as explained in the next slides. 

The n-gram database is built from a **10 millions words corpus** coming from Twitter, blogs and news websites. 

N-grams that occur **less than 10 times** in the corpus **are discarded**. This considerably reduces the size of the database while ensuring optimum results.




# N-GRAM MODEL

The n-gram model is explained nicely in this swiftkey [blog post](https://blog.swiftkey.com/neural-networks-a-meaningful-leap-for-mobile-typing/): 

<img style="width:60%;" src="https://sebastienplat.s3.amazonaws.com/f93f7de34fdf6b969404aab76eb0c4d31470386401999" alt="ngram"></img>




# STUPID BACKOFF

We want to calculate the ranking of as many words as possible, to ensure the **best possible prediction**. 

The [stupid backoff](http://www.aclweb.org/anthology/D07-1090.pdf) algorithm is used for sequences of words **not in the database** ("at the mall" in our example below): it looks at the word's ranking in a smaller sequence of words ("the mall") and applies a penalty to reflect its lower likelihood. 

In our example, "mall" is the most common word after "the", so it is ranked 1st. The penalty is applied and the final ranking of "mall" is 4th.

<img style="width:80%;" src="https://sebastienplat.s3.amazonaws.com/6986341cbbdcda15bf3dd63e402aa3f61470386565361" alt="stupidbackoff"></img>




# APP DESIGN

The app (available in the [demo](http://www.sebastienplat.com/smartKeyboard/demo)) mimicks the design of all existing smartphone design:

<img style="width:50%;" src="https://sebastienplat.s3.amazonaws.com/f19370419b2b2a4eeca68b02861779e11470387739555" alt="smatKeyboard"></img>

The app will update the suggestions depending on the cursor position, so the user can go back and easily change previously typed words.

