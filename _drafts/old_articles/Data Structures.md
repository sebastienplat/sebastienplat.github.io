# DATA STRUCTURES

#### Abstract Data Type

An **abstract data type** or **ADT**, is a logical description of how we view the data and the operations that are allowed.

The user only interacts with the **interface**. An interface only provides the list of  operations supported by the data structure, the type of parameters they can accept and the return type of these operations.

The **implementation**, also referred to as a **data structure**, is hidden one level deeper. It provides the internal representation of the data structure. Implementation also provides the definition of the algorithms used in the operations of the data structure.

![adt.png](https://sebastienplat.s3.amazonaws.com/301e388a0b1cefa0f33baa1f857a1ccb1487619035297)

By providing this level of abstraction, we are creating an **encapsulation** around the data: this **information hiding** provides an implementation-independent view of the data. 

It also allows the programmer to switch the details of the implementation without changing the way the user of the data interacts with it.


#### Performance of Python Data Structures

##### Lists

Two common lists operations are indexing and assigning to an index position. Both of these operations take the same amount of time no matter how large the list becomes: they are $$O(1)$$.

Another very common programming task is to grow a list:

+ the `append` method is $$O(1)$$
+ the concatenation operator `+` is $$O(k)$$, where $$k$$ is the size of the list that is being concatenated

Performance of lists operations can be found [here](http://interactivepython.org/runestone/static/pythonds/AlgorithmAnalysis/Lists.html#tbl-listbigo).

##### Dictionaries

Dictionaries differ from lists in that you can access items in a dictionary by a key rather than a position. 

All these operations are $$O(1)$$:

+ set / get key value
+ contains (in)

Performance of dictionaries operations can be found [here](http://interactivepython.org/runestone/static/pythonds/AlgorithmAnalysis/Dictionaries.html#tbl-dictbigo).

Checking if a key is in a dictionary is much faster than its list counterpart, which is $$O(n)$$.


#### Choice of Data Structure

Choosing a data structure largely depends on the frequency of the operations that needs to be performed:

+ Traversing
+ Searching
+ Insertion
+ Deletion
+ Sorting
+ Merging




# LINEAR STRUCTURES

Linear data structures are data collections whose items are ordered depending on how they are added or removed. Once an item is added, it stays in that position relative to the other elements that came before and came after it.

What distinguishes one linear structure from another is the way in which items are added and removed, in particular the location where these additions and removals occur. For example, a structure might allow new items to be added at only one end. Some structures might allow items to be removed from either end.


#### Stack

A [stack](http://interactivepython.org/runestone/static/pythonds/BasicDS/ImplementingaStackinPython.html) (sometimes called a “push-down stack”) is an ordered collection of items where the addition of new items and the removal of existing items **always takes place at the same end**. This end is commonly referred to as the “top.” The end opposite the top is known as the “base.”

Stacks are ordered **LIFO**. 

Stacks are very important data structures for the processing of language constructs in computer science. Almost any notation you can think of has some type of nested symbol that must be matched in a balanced order. 

Stacks are used to: 

+ convert decimal numbers to binary numbers [binary numbers](http://interactivepython.org/runestone/static/pythonds/BasicDS/ConvertingDecimalNumberstoBinaryNumbers.html)
+ calculate [complex arithmetic expression](http://interactivepython.org/runestone/static/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html) 


#### Queue

A [queue](http://interactivepython.org/runestone/static/pythonds/BasicDS/ImplementingaQueueinPython.html) is an ordered collection of items where the addition of new items happens at one end, called the “rear,” and the removal of existing items occurs at the other end, called the “front.” 

Stacks are ordered **FIFO**. 

OS use a number of different queues to control processes within a computer. The scheduling of what gets done next is typically based on a queuing algorithm that tries to execute programs as quickly as possible and serve as many users as it can.

#### Dequeue

A [deque](http://interactivepython.org/runestone/static/pythonds/BasicDS/ImplementingaDequeinPython.html), also known as a **double-ended queue**, is an ordered collection of items similar to the queue. It has two ends, a front and a rear, and the items remain positioned in the collection. 

What makes a deque different is the unrestrictive nature of adding and removing items. New items can be added at either the front or the rear. Likewise, existing items can be removed from either end. 

In a sense, this hybrid linear structure provides all the capabilities of stacks and queues in a single data structure.

_Note: using [collections.deque](https://docs.python.org/3/library/collections.html#collections.deque) while creating a Deque class will yield better performances)._

A deque can be used to solve the [palindrome problem](http://interactivepython.org/runestone/static/pythonds/BasicDS/PalindromeChecker.html).

#### List

A list is a collection of items where each item holds a relative position with respect to the others. More specifically, we will refer to this type of list as an **[unordered list](http://interactivepython.org/runestone/static/pythonds/BasicDS/ImplementinganUnorderedListLinkedLists.html#the-unordered-list-class)**. 

A linked list implementation maintains logical order without requiring physical storage requirements.

The basic building block for the linked list implementation is the **[node](http://interactivepython.org/runestone/static/pythonds/BasicDS/ImplementinganUnorderedListLinkedLists.html#the-node-class)**. Each node object must hold at least two pieces of information:

+ the list item itself (also called the data field)
+ a reference to the next node

The unordered list will be built from a collection of nodes, each linked to the next by explicit references. As long as we know where to find the first node (containing the first item), each item after that can be found by successively following the next links. 

Except for `add`, all methods are based on a technique known as **linked list traversal**: the process of systematically visiting each node.

To do this we use an external reference that starts at the first node in the list. As we visit each node, we move the reference to the next node by “traversing” the next reference.




# HASHING

#### Hashing

A hash table is a collection of items which are stored in such a way as to make it easy to find them later. 

Each position of the hash table, often called a **slot**, can hold an item and is named by an integer value starting at 0. Initially, the hash table contains no items so every slot is empty.

The mapping between an item and the slot where it belongs in the hash table is called the **hash function**. 

+ [Hashing](http://interactivepython.org/runestone/static/pythonds/SortSearch/Hashing.html)
+ [Bloom filters](http://stackoverflow.com/questions/4282375/what-is-the-advantage-to-using-bloom-filters)
+ [Compared performances of Maps ADT](http://interactivepython.org/runestone/static/pythonds/Trees/SummaryofMapADTImplementations.html): 
  + Sorted List 
  + Hash Table
  + Binary Search Tree 
  + AVL Tree
+ [BST vs Hash Tables](http://www.geeksforgeeks.org/advantages-of-bst-over-hash-table/)




