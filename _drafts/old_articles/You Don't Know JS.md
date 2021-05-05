# TERMINOLOGY

> This article in a synthesis of the excellent book series called "You don't know JS" from Kyle Simpson. It is available for free on [Github](https://github.com/getify/You-Dont-Know-JS). Many examples & text extracts have been taken verbatim.


#### Statemements

In a computer language, a group of words, numbers, and operators that performs a specific task is a **statement**. In JavaScript, a statement might look as follows:

```js
a = b * 2;
```

+ `a` and `b` are **variables**
+ `2` is a **litteral value**:  it is not stored in a variable
+ `=` and `*` are **operators**


Statements are made up of one or more expressions. An expression is any reference to a variable or value, or a set of variable(s) and value(s) combined with operators.

The statement `a = b * 2;` has four expressions:

| Expression |  Name | Action |
|:---------------:|:-----------:|:----------------|
| 2  				 | literal value expr. | - |
| b					 | variable expr.       | retrieve `b` current value |
| b * 2			   | arithmetic expr.   | do the multiplication          |
| a = b * 2     | assignment expr. | assign the result to the variable `a` |

A general expression that stands alone is called an **expression statement**. The following example is a **call expression statement**, as the entire statement is the function call expression itself:

```js
alert( a );
```


#### Operators

Operators are how we perform actions on variables and values. Here are some of the most common ones in JavaScript:

+ **assignment**: `=`
+ **math**: `+`, `-`, `*`, `/`
+ **compound assignment**: `+=`, `-=`, `*=`, `/=`
+ **increment/decrement**: `++`, `--`
+ **object property access**: `.` as in `console.log()`
+ **equality**: `==`, `===`, `!=`, `!==`
+ **comparizon**: `<`, `<=`, `>`, `>=`
+ **logical**: `&&`, `||`

_Note: The `=` equals operator is used for assignment -- we first calculate the value on the right-hand side (source value), then put it into the variable on the left-hand side (target variable)._


#### Types & Values

The different representations for values are called **types** in programming terminology. JavaScript has built-in types for each of these so-called **primitive values**: `number`, `string`, `boolean`. The other JS types are `object`, `null`, `undefined` and  `symbol`. 

_Note1: values that are included directly in the source code are called **literals**._  
_Note2: `typeof null` returns `object`._


#### Variables

The way variables are declared depend on the programming language.

+ **Static typing**: a variable holds a **specific type of value**
+ **Weak typing**: a variable can hold **any type of value** at any time

Static typing (also known as type enforcement) emphatizes program correctness, because it prevents unintended value conversions. 

Weak typing (also known as dynamic typing) emphatizes program flexibility, because  a single variable can represent a value even if it changes type during the code execution.

JavaScript uses dynamic typing: it has typed values, not typed variables. They must start with `a-zA-Z$_` and can also include `0-9`.

_Note: by convention, constant variables are written in uppercase with underscores._


#### Code Blocks

Code blocks are groups of several statements. In JavaScript, they are defined by wrapping one or more statements inside a curly-brace pair `{ .. }` and are typically attached to some other control statement like `if` or loops.

_Note: blocks are not required to end with a semicolon._


#### Conditionals 

The `if` statement requires an expression in between the parentheses ( ) that can be treated as either `true` or `false`. Coercion occurs if the expression does not evaluate to a boolean.

Other conditional are:

```js
// switch statement
switch (a) {
    case 2:
    case 10:
        // some cool stuff
        break;
    case 42:
        // other stuff
        break;
    default:
        // fallback
}
```

```js
// ternary operator
var a = 42;
var b = (a > 41) ? "hello" : "world";
```


#### Loops

Programming loops repeat a set of actions, typically in a block `{ .. }`, while a condition holds. The conditional is tested each time the loop block executes (ie. for each **iteration**).

_Note1: a loop can be stopped with JavaScript's `break` statement._  
_Note2: the `for` loop has three clauses: initialization, conditional, update._


#### Functions

Functions are a good way to break up the code's tasks into reusable pieces. They can include arguments (ie. parameters).

They can be declared as a **function variable** of the outer enclosing scope:

```js
function foo() {
    // ..
}
```

They can also be declared as an expression: 

```js
// anonymous function expression assigned to the foo var
var foo = function() {
    // ..
};

// named function expression
var x = function bar(){
    // ..
};
```




# COMPARING VALUES

#### Coercion

In JavaScript, a conversion between types is called **coercion**. It can be either **explicit**...:

```js
var a = "42";        // string type
var b = Number( a ); // coercion into number type
```

... Or **implicit**, for example when comparing values that are not already of the same type. A "loose equality" can be performed using the `==` operator.


#### Boolean coercion

When a non-boolean value is coerced to a boolean, it is evaluated as falsy or truthy:

+ **falsy values**: `0`, `-0`, `NaN`, `""`, `null`, `undefined`, `false` 
+ **truthy values**: all values not evaluated as falsy


#### Loose / Strict Equality

Here are a few rules for choosing `==` or `===` (full set of rules [here](http://www.ecma-international.org/ecma-262/5.1/)):

+ use `===` If either value could be a boolean or  `0`, `""`, `[]`
+ In all other cases, you're safe to use `==`

`array` are by default coerced to `string` by simply joining all the values with commas. You might think that two arrays with the same contents would be `==` equal, but they're not:

```js
var a = [1,2,3];
var b = [1,2,3];
var c = "1,2,3";

a == c;     // true
b == c;     // true
a == b;     // false
```

#### Inequality

Inequality can be used with `number` or `string` (alphabetical order). 

Coercion occur when one or both values are not `string`: both values are coerced into `number`, with value `NaN` when invalid, which is neither greater-than nor less-than any other value.

_Note: There are no "strict inequality" operators for inequality. _




# SCOPE

#### JavaScript Interpreter
  
A program must be translated into computer language before it can run. Depending on the programming language, it is done by using:

+ an **interpreter**: the translation is done from top to bottom, line by line, every time the program is run
+ a ** compiler**: the translation is done ahead of time. What is running is the already compiled computer instructions

JavaScript is interpreted. At run-time, the interpreter performs two passes:

1. **hoisting**: binds variable and function declarations to the scope
2. **code execution**: process function expressions and undeclared variables


#### Hoisting - Creating the scope

In JavaScript, a name enters a scope in one of four basic ways, in that order:

1. **language-defined**: all scopes are given `this` and `arguments`
2. **formal parameters**: `function foo( params );`
3. **function declarations**:  `function foo() {}`
4. **variable declarations**: `var foo;`

Function declarations and variable declarations are always moved (“hoisted”) invisibly to the top of their containing scope by the JavaScript interpreter. Function parameters and language-defined names are, obviously, already there.


#### Accessing the scope

**Scope** is the set of rules that determines where and how a variable (identifier) can be looked-up, for the purpose of:

+ **assigning** to the variable: LHS (left-hand-side) reference
+ **retrieving its value**: RHS (right-hand-side) reference

**Scopes are nested**: scope look-ups start in the currently executing scope, then go up the scope chain until either the variable is found or the outermost (aka, global) scope is reached.

They stop at  **first match**, even when the same identifier is used in an outer scope; it is called "shadowing" (the inner identifier "shadows" the outer identifier). 

In addition:

+ passing arguments to (assign to) function parameters is a LHS reference
+ Trying to use a variable outside its scope _(unfullfilled RHS references)_ will throw a `ReferenceError`
+ Trying to assign a value to an undeclared variable _(unfullfilled LHS references)_ will implicitely declare it globally, ie. in the top-level global scope
+ Using `"strict mode"` disallows the implicit auto-global variable declaration and returns `ReferenceError` instead

_Note: Global variables are also properties of the global object `window`, so they can be accessed with `window.foo` even when shadowing._




# FUNCTION VS BLOCK SCOPE

#### Function scope

In JavaScript, each function gets its **own scope**. No matter where a function is invoked from, or even how it is invoked, its lexical scope is only defined by where the function was declared.

It can be used to avoid collisions:

```js
function foo() {
	function bar(a) {
        i = 3; // i, declared in the parent scope, gets overwritten
        console.log( a + i );
    }
    
    // 'var i' is hoisted before the for...loop is called
    // this will return 3 first, then 11 indefinitely
    for (var i=0; i<10; i++) {
        bar( i * 2 ); // oops, infinite loop ahead!
    }
}

foo();
```

Declaring the variable `var i = 3;` in the `bar()` function will "shadow" the declaration made in its parent scope `foo()` and prevent the infinite loop.


#### IIFEs

Wrapping a function around a code snippet "hides" any enclosed variable or function declarations from the outside scope. Immediately Invoked Function Expressions _(ie. IIFEs)_ are well-suited for this purpose, because their name is not accessible outside their scope:

```js
var a = 2;

// IIFE creates a new scope - `IIFE` is bound only inside its own function
(function IIFE( global ) {

    var a = 3;
    console.log( a ); // 3
    console.log( global.a ); // 2

})( window ); // passing arguments

console.log( a ); // 2
```

The outer `( .. )` prevents the IIFE from being treated as a normal function declaration: it is interpreted as an expression instead.


#### Closure

A function always has access to its author-time lexical scope, even when called outside it. In other words, it always has a reference to the scope where it was created, and that reference is called **closure**.

In the following example, `add()`has a **lexical scope closure** over the inner scope of `makeAdder()` (nested scopes rule). It means that `add()`can access it at any later time.

```js
function makeAdder(x) {
    // parameter `x` is an inner variable

    // inner function `add()` uses `x`, so
    // it has a "closure" over it
    function add(y) {
        return y + x;
    };

    return add;
}

// `plusOne` gets a reference to inner `add(..)` 
// with closure over the `x` param of outer `makeAdder(..)`
var plusOne = makeAdder( 1 );
plusOne( 3 );       // 4  <-- 1 + 3
```

More practical example:

```js
function wait(message) {

    setTimeout( function timer(){
        console.log( message );
    }, 1000 );

}

wait( "Hello, closure!" );
```
The inner function `timer` is passed to `setTimeout(..)`. `timer` has a scope closure over the scope of `wait(..)`, keeping and using a reference to the variable `message`.


#### Loops & Closure

Executing the following code will return `6` five times, reflecting the final value of `i`. 

```js
for (var i=1; i<=5; i++) {
    setTimeout( function timer(){
        console.log( i );
    }, i*1000 );
}
```

This is because:

1. the `timeout` function callbacks are all running after loop completion
2. all callback functions are closed over the same global scope
3. when called, they refer to the same and only `var i`, that evaluates to `6`

To fix this behaviour, we need a new closured scope for each iteration of the loop. We can use an IIFE to create a new scope for each iteration:

```js
for (var i=1; i<=5; i++) {
	(function() {
        var j = i;  // needed, otherwise the IIFE inner scope is empty
    	setTimeout( function timer(){
        	console.log( j );
        }, j*1000 );
    })();
}
```

#### Loops & Block Scope

We can also use the new ES6 keywords `let` and `const`, that attach the variable declaration to the scope of the block they're contained in (commonly `{ .. }`). 

_Note1: Declarations made with `let` or `const` will not hoist: they will not "exist" in the block until the declaration statement._

_Note2: `const` values ares fixed. Trying to change them later will throw an error._


```js
var foo = true,
    baz = 10;

if (foo) {
    { // <-- explicit block
        console.log( bar ); // ReferenceError
        let bar = baz * 2;
        console.log( bar ); // 20
    }
}

console.log( bar ); // ReferenceError
```

Our loop example revisited:

```js
for (let i=1; i<=5; i++) {
    setTimeout( function timer(){
        console.log( i );
    }, i*1000 );
}
```

A `let` declaration used in the head of a for-loop is declared not just once for the loop, but for each iteration; it will be initialized with the value from the end of the previous one.

It means that each `timer`function will close over its own scope, which has a new instance of `i` with the expected value.


#### Modules

The most common usage of closure in JavaScript is the **module pattern**. Modules let you define private implementation details (variables, functions) that are hidden from the outside world, as well as a public API that is accessible from the outside.

```js
function CoolModule() {
    var something = "cool";
    var another = [1, 2, 3];

    function doSomething() {
        console.log( something );
    }

    function doAnother() {
        console.log( another.join( " ! " ) );
    }

    // public API object
    return {
        doSomething: doSomething,
        doAnother: doAnother
    };
}

// invoking the fn creates an instance of the module
var foo = CoolModule(); 

// these functions have closure over 
// the inner scope of the module "instance"
foo.doSomething(); // cool
foo.doAnother(); // 1 ! 2 ! 3
```

Modules are just functions, so they can receive parameters:

```js
function CoolModule(id) {
    function change() {
        // modifying the public API
        publicAPI.identify = identify2;
    }

    function identify1() {
        console.log( id );
    }

    function identify2() {
        console.log( id.toUpperCase() );
    }

    var publicAPI = {
        change: change,
        identify: identify1
    };

    return publicAPI;
}

// instance of the module
// with closure over passed parameter `id`
var foo1 = CoolModule( "foo 1" );
var foo2 = CoolModule( "foo 2" );

foo1.identify(); // "foo 1"
foo2.identify(); // "foo 2"
foo2.change();
foo2.identify(); // "FOO 2"
```

By retaining an inner reference to the public API object inside the module instance, we can modify it **from the inside**, including adding and removing methods, properties, and changing their values.




#  THIS

#### This

We saw earlier that `scope` is an **author-time binding**: it depends on where is has been declared.

`this` is the opposite: it is a **runtime binding**. It depends on where it has been called: the **call-site**.

More precisely: when a function is invoked, an activation record _(or execution context)_ is created. This record contains information about where the function was called from (the call-stack), how the function was invoked, what parameters were passed, etc. One of the properties of this record is the `this` reference.

_Note: it does not refer to the function itself._

```js
// `this` is the global object (or 'undefined' in strict mode)
function foo() { 
    console.log( this.bar ); 
}

var bar = "global";

// `this` is the `obj1` object
var obj1 = {
    bar: "obj1",
    foo: foo
};

var obj2 = {
    bar: "obj2"
};

// --------

foo();              // "global"
obj1.foo();         // "obj1"
foo.call( obj2 );   // "obj2" - `this` is the `obj2` object
new foo();          // undefined - `this` is a brand new empty object
```