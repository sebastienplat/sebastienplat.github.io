# OOP

#### First-Class everything 

> All objects that can be named in the language (e.g., integers, strings, functions, classes, modules, methods, and so on) have equal status. That is, they can be assigned to variables, placed in lists, stored in dictionaries, passed as arguments, and so forth.

To see the class of a Python object: 

```
type(obj)
```

**Classes** can be thought of as blueprints for creating objects. They're an instruction manual for constructing objects that share the same attributes and/or functions.

An object is an **instance** of its class.

#### Methods

A method differs from a function only in two aspects:

+ It belongs to a class, and it is defined within a class
+ Its first parameter has to be a reference to the instance which called the method.

_Note: This parameter is usually called _`self`_, but could also be called _`this`_ as in other languages._


##### Init method

The `__init__` method is called each time a new instance of a class is created. It is also called a class constructor.

It allows us to define the attributes of an instance (also called instance variables), which can be used by methods:

```python
class Robot:
 
  def __init__(self, name=None):
    self.name = name   
        
  def say_hi(self):
    if self.name:
      print("Hi, I am " + self.name)
    else:
      print("Hi, I am a robot without a name")
      
no_name = Robot()
no_name.say_hi() # Hi, I am a robot without a name

henry = Robot("Henry")
henry.say_hi() # Hi, I am Henry
```

##### Static methods

Static methods don't need a reference to an instance. They can be called via the class or the instance name, without the necessity of passing a reference to an instance to it.

They are created with the `@staticmethod` decorator syntax:

```
class Robot:
  __counter = 0
    
  def __init__(self):
    type(self).__counter += 1
        
  @staticmethod
  def RobotInstances():
    return Robot.__counter
  
print(Robot.RobotInstances()) # 0
x = Robot()
print(x.RobotInstances()) # 1
y = Robot()
print(x.RobotInstances()) # 2
print(Robot.RobotInstances()) # 2
```


##### Class methods

Class methods are not bound to instances but to a class. Their first parameter is a reference to a class object, and they can be called via an instance or the class name. 

Use cases:

+ factory methods
+ static methods calling other static methods

They are created with the `@classmethod` decorator syntax:

```
class fraction(object):

  def __init__(self, n, d):
    self.numerator, self.denominator = fraction.reduce(n, d)
  
  @staticmethod
  def gcd(a,b):
    while b != 0:
      a, b = b, a%b
    return a

  @classmethod
  def reduce(cls, n1, n2):
    g = cls.gcd(n1, n2)
    return (n1 // g, n2 // g)

  def show_result(self):
    return str(self.numerator)+'/'+str(self.denominator)
    
x = fraction(8,24) 
x.show_result()  # 1/3
```


#### Attributes & Properties

##### Public / Protected / Private

Class attributes can either be:

| Naming | Type      | Meaning   |
|--------|-----------|-----------|
| name   | Public      | can be freely used inside or outside of a class definition     |
| _name  | Protected | should not be used outside of the class definition, unless inside of a subclass definition |
| __name | Private    | inaccessible and invisible, except inside of the class definition itself      |


##### Getter / Setter

Getters and setters, _aka_ mutator methods, are used in many object oriented programming languages to ensure the principle of data encapsulation.

Python uses properties to define getters & setters:

```
class P:

  def __init__(self, x):
    self.ourAtt = x

  @property # getter method
  def ourAtt(self):
    return self.__ourAtt # private attribute ourAtt

  @ourAtt.setter  # setter method for property function ourAtt
  def ourAtt(self, val):
    if x < 0:
      self.__ourAtt = 0
    elif x > 1000:
      self.__ourAtt = 1000
    else:
      self.__ourAtt = val
```

Another example without an  explicit setter method:

```
class Robot:

  def __init__(self, name, build_year, lk = 0.5, lp = 0.5 ):
    self.name = name
    self.build_year = build_year
    self.__potential_physical = lk
    self.__potential_psychic = lp

  @property
  def condition(self):
    s = self.__potential_physical + self.__potential_psychic
    if s <= -1:
      return "I feel miserable!"
    elif s <= 0:
      return "I feel bad!"
    elif s <= 0.5:
      return "Could be worse!"
    elif s <= 1:
      return "Seems to be okay!"
    else:
      return "Great!" 
  
if __name__ == "__main__":
  x = Robot("Marvin", 1979, 0.2, 0.4 )
  y = Robot("Caliban", 1993, -0.4, 0.3)
  print(x.condition) # Seems to be okay!
  print(y.condition) # I feel bad!
```

##### Class Variables

Class variables are shared by all instances of a class. For example, the rating of movies in the US:

```
class Movie:
  valid_ratings = ['G', 'PG', 'PG-13', 'R']
```

There are also a few [predefined class variables](https://docs.python.org/3.5/reference/datamodel.html). For example, the class documentation string `__doc__`:

```
class Movie:
  """Stores movie-related info"""
  
print(Movie.__doc__) # Stores movie-related info
```

Other common examples: 

  + `__name__`: The class name
  + `__module__`: The name of the module the class was defined in

#### Inheritance

Classes can be built from existing classes. They have the same attributes and methods as their parent class, but also their own, more specific ones.

Example: 

```
# parent class: pets
class Pets:
  name = "pet animals"

  @classmethod
  def about(cls):
    print("This class is about {}!".format(cls.name))
  
p = Pets()
p.about() # This class is about pet animals!

# Child class: Cats & Dogs
class Cats(Pets): # inherits from Pets
  name = "cats"
  
class Dogs(Pets): # inherits from Pets
  name = "'man's best friends' (Frederick II)"

c = Cats()
c.about() # This class is about cats!

d = Dogs()
d.about() # This class is about man's best friends' (Frederick II)!
```

Another, more complex, example:

```
# parent class
class Parent:
  def __init__(self, last_name, eye_color): 
    self.last_name = last_name
    self.eye_color = eye_color

# Child class
class Child(Parent): # inherits from Parent
  def __init__(self, last_name, eye_color, number_of_toys): 
    Parent.__init__(self, last_name, eye_color)
    self.number_of_toys = number_of_toys
```

#### Magic Methods

Magic methods, _aka_ double underscore or dunder methods, are "magic" because you do not have to invoke them directly. 

The `__init__` function is a magic method.

The `__str__` method is best-suited for end users output. It is called via `str` and `print`:

```python
class A:
  def __str__(self):
    return "42"

a = A()

print(str(a))
> 42
print(a)
> 42
```

The `__repr__` method is best-suited for the internal representation of an object. It is useful for debug purposes, as you can decide exactly what string is returned:

```python
class Foo(object):
  def __init__(self, bar):
    self.bar = bar
  def __repr__(self):
    return "<Foo bar:%s>" % self.bar
    
repr(Foo(42))
> "<Foo bar:42>"
```

When possible, the output of `__repr__` should be a string which can be parsed by the python interpreter to return an equal object:

```
o == eval(repr(o))
```

This way, `__repr__` outputs can easily be copy-pasted into a Python session to recreate the object.

#### Further readings

+ [Overview of classes and OOP](https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/)
+ [Usefulness of OOP via an RPG example](http://inventwithpython.com/blog/2014/12/02/why-is-object-oriented-programming-useful-with-an-role-playing-game-example/)
+ [When to use OOP](https://www.packtpub.com/books/content/python-3-when-use-object-oriented-programming)
+ [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)


