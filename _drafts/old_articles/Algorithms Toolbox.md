# INTRODUCTION

#### Overview

_Note: the source material for this post comes from Khan Academy. It can be found [here](https://www.khanacademy.org/computing/computer-science/algorithms)._

As shown in the Fibonacci example, choosing the right algorithm can dramatically improve the time needed to solve a problem. But doing so often requires knowledge about the problem *(see GCD calculation and the [Euclid algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm))*.

There are three levels of algorithm design:

+ **naive**: translate problem definition to algorithm. Slow.
+ **standard**: apply standard tools/techniques
+ **optimized**: improve existing algorithm (reduce runtime)
+ **magic**: unique insights


An algorithm is a compromise between correctness & efficiency. Its efficiency is measured by using __asymptotic analytics__.


#### Asymptotic Notations

The running time of an algorithm depends on its **growth rate**: how fast a function grows with the input size. Keeping only the most significant term & dropping cofficients is called the **asymptotic notation**: it approximates the running time for **large input sizes**.

There are several asymptotic notations:

+ **big $$\Theta$$**: asymptotic tight bound (ie. both lower & upper bounds)
+ **big $$\Omega$$**: asymptotic lower bound
+ **big O**: asymptotic upper bound

The **big O** is the most commonly used for **growth rates comparizon**.

<hr>

A few examples of frequent asymptotic functions, from slowest to fastest growing:

$$1 < log(n) < n < n\ log(n) < n^2 < 2^n$$

_Note: $$log(n)$$ growth rate can occur in recursive functions, where each iteration splits the problem in two roughly equal subproblems._

_Note: exponential functions $$a^n$$, where $$a > 1$$, grows faster than polynomial function $$n^b$$ , where $$b$$ is any constant._


#### Iterative vs Recursive Calls

An iterative algorithm is usually faster than a recursive one, when all subproblems must be calculated, as there is less overhead. But recursive calls are made only when useful for the final solution, so it will be best suited for situations when not all the subproblems are needed.


#### Stress Testing

Testing should cover:

+ small, manual preliminary tests
+ each possible **type of answer** (smallest, biggest, no answer, etc.)
+ **time/memory limit** for large inputs
+ **corner cases**: duplicates in the input, empty set, tree with only of chain of nodes, etc.

If these tests are not enough, it is possible to use **stress testing** to find examples where the algorithm fails. It requires:

+ the algorithm to investigate
+ another algorithm that always produces the correct output, even if very slow
+ a test generator
+ an infinite loop comparing the two algorithms for each generated test




# FIBONACCI NUMBERS

#### Definition

The $$n^{th}$$ Fibonacci Number is calculated as follows:

$$F_n = \begin{cases} 0, & n=0, \\ 1, & n=1, \\ F_{n-1} + F_{n-2}, & n>1 \end{cases}$$

They grow **exponentially**: $$F_{100}$$ has already 21 digits, $$F_{500}$$ more than 100. 

#### Naive algorithm

We can calculate the Fibonacci numbers recursively, as described in the definition:

```js
function fibRecurs (n) {
  if (n <= 1) return n;
  return fibRecurs (n-1) + fibRecurs (n-2);
}
```

But this is very inefficient. Let $$T(n)$$ denote the lines of code executed by fibRecurs(n).

$$T(n) = \begin{cases} 2, & n \leq 1, \\ T(n-1) + T(n-2) + 2, & n>1 \end{cases}$$

This is roughly the Fibonacci number itself !


#### Efficient Algorithm

In the previous algorithm, each Fibonacci number is calculated several times. A better solution is to calculate them only once & save them in an array:

```js
function fibList (n) {
  var F = [0,1]; // first two Fibonacci numbers
  for (let i=2; i<=n; i++) F.push(F[i-1] + F[i-2]);
  return F[n];
}
```
This time, $$T(n) = 2 \times (n-1) + 2$$, which is easy to compute even for large values of $$n$$.

More specifically, in terms of big O notation:

$$F_{runtime} = \begin{cases} var\ F..., & O(1), \\ for...\ loop, & loop\ O(n)\ times \\ \  \ F.push, & O(n) \\ return\ F[n], & O(1) \end{cases}$$

_Note: F.push is O(n) because additions runtime is proportional to the number of digits, which in this case is proportional to $$n$$._

So: $$F_{runtime} =  O(1)+ O(n) \times O(n)+ O(n)+ O(1) =  O(n^2)$$.




# GREEDY ALGORITHM

#### Example

```js
// stops include start at pos [0] and destination at pos [n+1]
var minRefills = function (stops, tankDist) {
  var numRefills = 0, n = stops.length - 2; // pos of last gas station
  var pos = 0; // pos will identify were we are on the road
  while (pos <= n) {
    lastRefillPos = pos;
    // we continue until we reach the end or the next gas station
    while (pos <= n && stops[pos+1] - stops[lastRefillPos] <= tankDist) {
      pos++;
    }
    // if the next stop is too far away, we cannot move
    if (pos === lastRefillPos) return "IMPOSSIBLE"; 
    // if we are not yet at the end, we refill the tank
    if (pos <= n)  numRefills++;
  }
  return numRefills;
};
```




# DIVIDE & CONQUER

#### Principle

The divide & conquer method consists of three steps:

1. **Divide** a problem into smaller sub-problems
2. **Conquer** each sub-problems
3. **Combine** the solutions to solve the problem

_See merge sort / quicksort for for in-depth examples._

The running time of a divide & conquer algorithm can be calculated using the **Master Theorem** *([link](https://en.wikipedia.org/wiki/Master_theorem))*.

#### Recursion

In practice, divide & conquer is often implemented through regression. A recursive algorithm must follow two important rules;

1. each recursive call should be on a smaller instance of the same problem
2. the recursive calls must eventually reach a base case, which is solved without further recursion

#### Memoization

In order to speed up recursive algorithms, it is often a good idea to store results after they are calculated once, and reuse the value at every subsequent call.


#### Factorial

$$n!$$ is useful to count how many different ways we can arrange $$n$$ things, or how many ways we can choose $$k$$ things from a collection of $$n$$ things: $$n! / (k! \times (n-k)!)$$ ways to choose.

Factorials can be computed recursively by noting that $$n! = n \times (n-1)!$$, with $$0!=1$$.

```js
var factorial = function(n) {
  if (n===0) return 1;
  else if (n>0) return n * factorial (n-1);
  else return;
}; 
```

#### Palindrome

Checking if a string is a palindrome can also be done recursively:

```js
var firstChar = function(str) { return str.slice(0, 1); };
var lastChar  = function(str) { return str.slice(-1); };
var midChars  = function(str) { return str.slice(1, -1); };

var isPalindrome = function(string) {
  if (str.length<=1) return true;
  else if (firstChar(str) === lastChar(str)) return isPalindrome(midChars(str));
  else return false;
};
```

#### Tower of Hanoi

The Tower of Hanoi problem can also be solved recursively, for any number of disks to be moved from any peg to another.

```js
var solveHanoi = function(numDisks, fromPeg, toPeg) {
  if (numDisks === 0) return true;
  var sparePeg = getSparePeg(fromPeg, toPeg);
  solveHanoi(numDisks-1,fromPeg,sparePeg); // move all previous disks to the spare peg
  moveDisk(fromPeg, toPeg); // move the bottom disk to the final peg
  solveHanoi(numDisks-1,sparePeg,toPeg); // move all previous disks to the final peg
};
```

Applying the algorithm, we can calculate the number of steps required to move $$n$$ disks from one peg to another:

+ two disks: 1 + 1 + 1 = 3 steps
+ three disks: 3 + 1 + 3 = 7 steps
+ four disks: 7 + 1 + 7 steps
+ ...
+ $$n$$ disks: $$2^n - 1$$ steps

This can be proved recursively:

$$s_{n} = 2 \times s_{n-1} + 1 = 2 \times (2^{n-1} - 1) + 1 = 2^n - 1$$





# SORTING

#### Compared Performance

_Note: animations for many several algorithms can be found [here](https://www.toptal.com/developers/sorting-algorithms/)._

Expressed as **big $$\Theta$$**: asymptotic tight bound.

| Algorithm | Worst case | Average   | Best Case |
|:-------------:|:--------------:|:--------------:|:----------:|
| Selection   |    $$n^2$$     |  $$n^2$$  |  $$n^2$$  |
| Insertion    |    $$n^2$$      | $$n^2$$  |  $$n$$  |
| Quicksort  |    $$n^2$$      |  $$n\ lg\ n$$ |  $$n\ lg\ n$$ |
| Merge         |   $$n\ lg\ n$$  |  $$n\ lg\ n$$ |  $$n\ lg\ n$$ |


#### Selection Sort

Selection Sort is a sorting algorithm that works by repeatedly swapping values in the correct position.

```js
var selectionSort = function(array) {
  var minIndex;
  for (var i=0; i<array.length; i++) {
    minIndex = indexOfMinimum(array, i); // identify index of minimum value, from position i
    swap(array, i, minIndex); // swap minimum value & value at position i
  }
};
```

The total running time for selection sort has three parts (each call a an action takes constant time):

1.  $$n$$ loop calls
2. $$n \times (n+1) / 2$$ loop calls inside indexOfMinimum _(sum of 1 to $$n$$)_
3. $$n$$ calls to swap

It means the **running time of Selection Sort** is $$\Theta (n^2)$$.


#### Insertion Sort

Insertion Sort is a sorting algorithm that works by successively inserting new values to the correct position.

```js
// add value to already sorted subarray 0...rightIndex
var insert = function(array, rightIndex, value) {
  for(var i = rightIndex; i >= 0 && array[i] > value; i--) { //note the two conditions
    array[i + 1] = array[i];
  }   
  array[i + 1] = value;
};

// loop over increasingly larger subarrays
var insertionSort = function(array) {
  for(var j=1; j<array.length; j++) {
    insert(array, j-1, array[j]);
  }
};
```

The total running time for insertion sort depends on the initial state of the array. The shorter the insert loop (ie. few indexes where array[i] > value), the faster the algorithm will perform.

1. $$n-1$$ loop calls
2. min $$0$$ & max $$ n \times (n-1) /2$$ loop calls inside insert loops _(sum of 1 to $$n-1$$)_
3. $$n-1$$ calls to udpate value

The  **running time of Selection Sort** is **at best** $$\Theta (n)$$ and **at worst** $$\Theta (n^2)$$.


#### Merge Sort

The mergeSort algorithm recursively splits the array in two equal subarrays to sort them, then merge them and sort the resulting array.

```js
var mergeSort = function(array) {
  
  if (array.length === 1) return array; // base case
  
  // keep splitting until array of size 1
  var q = Math.floor(0.5*array.length);
  var lowHalf  = mergeSort(array.slice(0,q)); // sort the left-side array
  var highHalf = mergeSort(array.slice(q));   // sort the right-side array
  return merge(lowHalf, highHalf);
  
};

var merge = function (lowHalf, highHalf) {
  var sortedArray = [];
  // while loop until one of the two subarrays is empty
  while (lowHalf.length > 0 && highHalf.length > 0) {
    if (lowHalf[0] < highHalf[0]) {
      sortedArray.push(lowHalf.splice(0,1));
    }
    else {
      sortedArray.push(highHalf.splice(0,1));
    }
  }
  // add the remaining array & return
  return sortedArray.concat((lowHalf.length>0) ? lowHalf: highHalf);
};
```

The merging of $$n$$ elements split in two sorted subarrays is $$\Theta(n)$$:

1. $$n$$ operations to copy each element into either lowHalf or highHalf
2. $$n$$ loops to populate the final array; for each loop:
  + comparizon of two elements _(the first of lowHalf and highHalf)_
  + copy the smallest one back into the array
  + when either one of the two subarrays is empty, copy the other one back into the array
 
Let's say the merging time is $$c \times n$$. The total running time is $$c \times n$$, plus the time taken by the two recursive calls on $$n/2$$ elements. Each of these has a merge step twice as fast, so their running time is $$2 \times (c \times n/2) = c \times n$$ plus the time taken by the four recursives calls on $$n/4$$ elements.

It means that the total running time of the algorithm is $$l \times cn$$, where $$l$$ is the number of iterations until the remaining subproblems are arrays of size 1: $$l = lg\ n +1$$. 

So **the total running time of the mergeSort algorithm** is $$\Theta(n\ lg\ n)$$.


#### Quicksort

QuickSort recursively chooses a pivot value & move it to its final position, while reordering the other values so that lower elements are to its left & greater elements to its right. It then repeats the process for the lower & greater subarrays.

In the following example, the pivot is chosen as the rightmost one. It could also be chosen at random.

```js
var swap = function(array, i, j) {
	var temp=array[i];
    array[i] = array[j];
    array[j] = temp;
};

// position pivot & reorganize lower+greater values
var partition = function(array, l, r) {
  var q = l,
      pivot = array[r];
  for (var j = l; j < r; j++) {
    if (array[j] <= pivot) {
      swap(array,j,q);
      q++;
    }
  }
  swap(array,q,r);
  return q;
};

var quickSort = function(array, l, r) {
  // keep splitting until array of size 1
  if (l < r) {
    var q = partition(array, l, r); // fix the pivot position
    quickSort(array, l, q-1); // sort the left-side array
    quickSort(array, q+1, r); // sort the right-side array
  }
  return array;
};
```

In the **worst-case** scenario (array already sorted, for example), each partition returns the entire array minus its pivot. In that case, similar to the Selection Sort, **the total running time of the QuickSort algorithm** is $$\Theta(n^2)$$.

In the **best-case** scenario, each partition returns two half-size subarrays. In that case, similar to Merge Sort, **the total running time of the QuickSort algorithm** is $$\Theta(n\ lg\ n)$$.




# KNAPSACK PROBLEM

#### Overview

The knapsack problem aims to choose items to fill a knapsack, such as:

+ the total items weight is **below the max capacity** $$W$$
+ the total items value is **maximum**

Each item to choose from has a weight $$w_i$$ and a value $$v_i$$. The optimal algorithm depends on certain conditions regarding the items.


#### Greedy 

The **greedy algorithm** is optimal when we can choose **all or part of each item**, otherwise known as **fractional knapsack**.

+ **greedy choice**: start with the item with max value per unit weight, and use as much of it as possible
+ **safe move**: if there was an optimal solution that do not use as much of this item as possible, we could increase the sack's value by using all of it, contradicting our statement

```js
// items is an array of [itemTotalWeight, itemUnitValue] sorted by unit value
var knapSack = function (W, items) {
  
  var sackValue = 0,
      weights = Array(items.length).fill(null);
      
  for (var i=0, l=items.length; i<l; i++) {
    if (W === 0) return [weights, sackValue];
    itemWeight = Math.min (W, items[i][0]); // we use the max we can
    sackValue = sackValue + itemWeight * items[i][1];
    weights[i] = itemWeight;
    W = W - itemWeight;
  }
  return [weights, sackValue];

};
```

#### Dynamic Algorithm

The greedy algorithm will fail for the **discrete knapsack**, where an item can be taken in full or not at all (no fraction allowed), because it will not use the full knapsack capacity. So we will use a **dynamic algorithm** instead.

##### Unlimited Items

In this version of the problem, there is an **unlimited quantity** of each item. Let's consider an optimal solution. When we remove an item $$i$$, we get an optimal solution for $$W - w_i$$ (cut & paste trick).

_Note: if the solution was not optimal, we could substitute the items in the sack by ones with higher values, which would contradict our hypothesis._

It means the optimal solution is $$value(W) = max_i \left\{  (value(W - w_i) + v_i \right\} $$


```js
// items is an array of [itemWeight, itemValue]
var knapSack = function (W, items) {

  var sackValue = [0], // value for capa = 0
      tmpValue = 0;

  // we loop through increasing sack capacity
  for (var w=1;  w<=W; w++) {
    sackValue.push(0); // we initialize max value for w
    // we look for the best combo previous max / item 
    for (var i=0, l=items.length; i<l; i++) {
      var itemWeight = items[i][0], 
          itemValue  = items[i][1];
      if (itemWeight <= w) {
        tmpValue = sackValue[w - itemWeight] + itemValue;
        if (tmpValue > sackValue[w])  sackValue[w] = tmpValue;
      }
    }
  }
  return sackValue;

};
```

The running time of this algorithm is $$O(n \times W)$$.


##### Limited Items

In this version of the problem, each item can be used **at most once**. Our previous hypothesis did not assume that, so we need to reformulate.

Let's consider an optimal solution. It can either include the last item of the items list, or not. 

+ if yes: removing it gives us an optimal solution for $$W - w_n$$, with a smaller list of items $$1$$ to $$n-1$$ (as each item can be used at least once)
+ if not: the problem is the same, but with a smaller list of items $$1$$ to $$n-1$$

It means the optimal solution is:

$$value(W, i) = max_i \left\{  (value(W - w_i, i-1) + v_i,\  (value(W, i-1) \right\} $$


```js
// items is an array of [itemWeight, itemValue]
var knapSack = function (W, items) {
	
  var sackValue = [],
      tmpValue = 0;
			
  sackValue.push(Array(items.length).fill(0)); // capa = 0
  
  for (let w=1; w<W; w++) {
    sackValue.push([]); // values for w
    for (let i=0, l=items.length; i<l; i++) {
      sackValue[w].push([]); // max value for w, items 1 to i
			
	  // first hyp: i is not in the optimal solution
	  sackValue[w][i] = i>0 ? sackValue[w][i-1] : 0;
			
	  // second hyp: i is in the optimal solution 
      // (ie. can we improve the solution by including i) 
	  if (items[i][0] <= w) {
	    tmpValue = (i>0 ? sackValue[w - items[i][0]][i-1] : 0) + items[i][1];
		if (tmpValue > sackValue[w][i]) sackValue[w][i] = tmpValue;
	  }	
    }	
  }
  return sackValue;
};
```

The running time of this algorithm is $$O(n \times W)$$.

To reconstruct an optimal solution, we must backtrace the path taken by the algorithm to reach it, and identify by a boolean whether an item was used or not.




# OTHER ALGORITHMS

#### Binary Search

Binary search is an efficient algorithm for **finding an item from an ordered list**. It works by repeatedly dividing in half the portion of the list that could contain the item. 

For an array $$myArray$$ of n items:

1. Let $$min = 0$$ and $$max = n-1$$
2. If $$max < min$$, then stop: $$target$$ is not present in $$myArray$$. Return -1.
3. Calculate the mean of $$min$$ & $$max$$, rounded down so it is an integer
4. If $$myArray[mean] = target$$, stop. You found it!
5. If the guess was too low, set $$min = guess + 1$$, ie. ditch the lower half
6. If the guess was too high, set $$max = guess - 1$$, ie. ditch the upper half
7. Go back to step two

Binary search is $$O(log\ n)$$.
