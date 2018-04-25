# I want to do "X"

The following is a guide for how to get started with programming side projects. If something's missing and you know about it, feel free to submit a pull request and add it.

- [I want to do "X"](#i-want-to-do-x)
  * [Mobile Apps](#mobile-apps)
  * [Linux](#linux)
  * [Machine Learning](#machine-learning)
    + [Overview](#overview)
    + [First Steps](#first-steps)
    + [First Project](#first-project)
    + [Step 1](#step-1)
    + [Step 2](#step-2)
    + [Step 3](#step-3)
    + [Step 4](#step-4)
  * [Step 5](#step-5)
  * [Step 6](#step-6)
  * [Security](#security)
    + [Overview](#overview-1)
    + [First Steps](#first-steps-1)
    + [First Project](#first-project-1)
      - [Step 1](#step-1-1)
      - [Step 2](#step-2-1)
      - [Step 3](#step-3-1)
    + [Step 4](#step-4-1)
    + [What else?](#what-else)
    + [Projects](#projects)
  * [Desktop apps](#desktop-apps)
  * [Video Games](#video-games)
    + [I want to be a better programmer](#i-want-to-be-a-better-programmer)
      - [Pick an Engine](#pick-an-engine)
      - [A First Project](#a-first-project)
    + [I want to make really good games](#i-want-to-make-really-good-games)
    + [Unity](#unity)
      - [A First Project](#a-first-project-1)
  * [Frontend Web](#frontend-web)
    + [First Steps - HTML](#first-steps---html)
      - [Adding style - CSS](#adding-style---css)
      - [Making it do things - Javascript](#making-it-do-things---javascript)
  * [Backend Web](#backend-web)
  * [Infastructure and AWS](#infastructure-and-aws)
  * [Building Programming Languages](#building-programming-languages)
    + [Overview](#overview-2)
    + [First Project](#first-project-2)
      - [Step 1: Write a grammar](#step-1-write-a-grammar)
      - [Step 2: Write a lexer](#step-2-write-a-lexer)
      - [Step 3: Write a parser](#step-3-write-a-parser)
      - [Step 4: Build an interpreter](#step-4-build-an-interpreter)
    + [Where to go from here](#where-to-go-from-here)
      - [Compiler Compilers!](#compiler-compilers)
      - [Formal grammars](#formal-grammars)
    + [Example projects](#example-projects)
      - [A Portfolio Language](#a-portfolio-language)
      - [Code Generators](#code-generators)
      - [AST Formatter](#ast-formatter)
      - [Build a simple scripting language](#build-a-simple-scripting-language)
  * [Learning specific programming languages](#learning-specific-programming-languages)
    + [Python](#python)
      - [First Program](#first-program)
      - [Basic Python](#basic-python)
      - [Basic Constructs](#basic-constructs)
      - [Using Python!](#using-python)
      - [Deeper](#deeper)
      - [Miscellaneous](#miscellaneous)
    + [Ruby](#ruby)
    + [C++](#c)
  * [Functional Programming](#functional-programming)

## Mobile Apps

//TODO

## Linux

//TODO

## Machine Learning

### Overview
Machine learning is the act of teaching a computer to do something that cannot be solved with hard coded solutions. The big thing right now is machine learning! Self-driving cars, figuring out if a person has cancer... Machine learning is used in everything! But, it's really not that scary! We promise! There are some amazing libraries that make this very simple.  
Okay, I lied a little bit. The math behind machine learning algorithms is quite scary at times. But, using the algorithms themselves are not that hard.

### First Steps
Picking a problem. Do you want to see how accurate you can predict the type of a cat based upon it's age, color, paw size and weight?   This can be done!

### First Project
Logistic regression is just like linear regression, but instead of Mx+b, it solving for a more advanced problem. We will be trying to predict the survivability of people on the [Titanic](https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv) based upon Sex, Age, Fare and many more features. The data set can be found in a link above. 
It should be known which libraries are used in this model:  
```  
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from patsy import dmatrices
```
These will be talked about later. Install these using pip, the python package installer. 

### Step 1
The first step to doing any sort of machine learning algorithm is getting data for this first. There are a wide range of [data sets](http://archive.ics.uci.edu/ml/datasets.html) publicly available on the Internet, ready to be used.  
But, you can write a web scraper to take information from Wikipedia, Google or hold polls to get your information if you cannot find what you're looking for. Our data has already been made and formatted for us luckily!


### Step 2
- [Pick an algorithm to use](https://blog.statsbot.co/machine-learning-algorithms-183cc73197c?gi=cdcbc147c7a0):
At a high level, there are two types of algorithms: supervised and unsupervised. *Supervised* means to give the inputs and outputs to the set, hoping to come up with a function that can represent the data correctly. This is quite more common; support vector machines, neural nets and much,much more. *Unsupervised* means to give the inputs to the set, hoping to learn something about the data. This will *create* the outputs for you. An example of this is MeanShift algorithm, which tries to find the amount of different clusters are in the dataset.
- Don't be afraid:  
It's okay to try out multiple algorithms for your problem! You can try them all if you want!  
For our project we'll be using Logistic regression to predict the model.

### Step 3
Learn the algorithm/how to do it:  
There are two things that can be done here: Learn the mathematics and complete understanding of the algorithm or learn how to use the algorithm at a level, which is just understanding how it works. This will just go through the implementation of the algorithms, rather than the math. But, these are out there! So, if you're looking for a deeper understanding then good luck!  

There are several libraries that can be used for this sort of thing. [Sklearn](http://scikit-learn.org/) has implementation for practically any algorithm that you could find, that is quite easy to use.  
For [neural nets](http://neuralnetworksanddeeplearning.com/chap1.html) has a fantastic implementation of a neural net AND dives deep into the mathematics.  
[Andrew Ng](https://www.coursera.org/learn/machine-learning) has a fantastic series on machine learning in general.  
- Short list of machine learning algorithms that are common
    - Neural Networks(deep, convolutional, modular)  
    Great for representing data that cannot easily be easily mapped.
    - Support Vector Machines:  
    A sequence classifier; practically just sets a line into the data.
    - Linear Regression:   
    Finds a linear representation of the data. Note: Think Mx+b. This just finds a representation of the data that's a linear line. It can be done for more than one input also.
    - K-nearest Neighbor:  
    An algorithm for clustering data. It will group it into k groups.  
    - Logistic Regression:  
    Finds a sigmoid function represented version of the data.

### Step 4
Converting the data into a form that the algorithm can understand:  
Very, very, very important! How is a picture represented? A 28 by 28 picture will be represented binary values of 0's and 1's depending on the brightness of the picture. This will be 784 inputs into the neural network.  

- [Representing Data](https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/):   
This is a good article about transforming data in the right way.  
- [Text Classification](https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6?gi=35cadf2889a3):  
Representing Twitter would be great! But, how does this work? That's a great question! This shows off to represent the information.  
- Our project already has most of the data formatted how we would like. But, there are a few issues with some data points. For instance, does passengerID really have anything to do with the accuracy of the model? No, not at all. So, dropping features that don't pertain to the actual result are important. We will drop passengerID, Ticket, Name and Cabin from the Titanic dataset.

A few standards in the ML land:  
- Python is great for machine learning algorithms!
- Data is held in **csv** files typically. These are huge sorts of data, that are represented as rows for the data point and columns for each feature.
- Pandas **dataframes** are how the data is stored inside of Python. They are great for manipulating the data put into the dataframe.
- **Numpy** is the most common way to store a matrix. Use this or beware of the consequences.

For getting the data in Python use pandas! The function below reads the Titanic data as a csv file into the dataframe:
```
def get_data():
    df = pd.read_csv("https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv")
    return df
```

To drop the columns from the table, we use the tables of the columns in a similar command.
```
    df = get_data()
    data = df.drop(columns = ['PassengerId','Ticket','Name','Cabin'])
```

These aren't cold cut things that need to be used, but they're very nifty to work with!

## Step 5
Run the tests! So, this has a few particular ways it's supposed to be done:   
- Split the data into a training set and testing set. The training set is for teaching the model; while the testing set is for checking for the accuracy of it.  
- Train the model:  
This looks like running all of the training data through the machine learning algorithm itself.  
- Test the model:  
Give the trained algorithm a set of inputs, without the output. After this, check to see how many the model accurately predicted.  
- Try again!:  
Don't just run the model once; run it at least three times in order to understand the actual accuracy of the model.
- Don't overfit:  
The term 'overfitting' means to run the algorithm over so many times on the training set, that the model will ONLY be correct for the training set. In practice, don't run the algorithm through too many iterations. This sounds like a great idea but the output from the testing set and other unknown values will yield bad results.  

Code for using statsmodels to create the model:  
    ```
    y, X = dmatrices("Survived ~ Pclass+Sex+Age+SibSp+Parch+Fare+Embarked", df, return_type = 'dataframe')
    ```
Training and testing the model:
```
    logit = sm.Logit(y,X)
    result = logit.fit()
    print result.summary2()
```

## Step 6
How was the accuracy? It probably came out to be very bad... but, that's okay! Try with less features, give it more or less data points, try a different machine learning algorithm entirely! Mr Beaver, who works as NextIT on systems trying to understand context of spoken language, claimed who would test 10 different ML algorithms to see which one worked the best for him. So, enjoy the endless amount of possibilities in the field!



## Security
### Overview
Security itself is a very broad topic, as it encapsulates every app, website, backend or anything else very created that can be seen. Understanding basic security concepts for different types of applications is essential to make a good app. For instance, when using a backend database it's very important to use prepared statements with parameters. Otherwise, information could be stolen from the database, which makes a frowny face.

### First Steps
Read!  
REad!  
REAd!  
READ!  
Listen!  
LISTEN!  
Everyone starts at the bottom of the totem pole; but we've all got to get started at some point!  
Great resources to read/listen from:  

- [Security Now](https://twit.tv/shows/security-now), is the best security podcast that discusses anything from updates in https, TLS and other things to recent hacks that have happened in the industry. Steve Gibson hosts this, with Leo creates more dialog. For a security podcast, it's quite entertaining!  
- [Krebs On Security](https://krebsonsecurity.com/), is one of the best blogs that a person can stay up-to-date with. He's the first to know about security breaches and other things in the industry. He usually has great posts about every 3 days on something new.  
- [List of Blogs](https://heimdalsecurity.com/blog/best-internet-security-blogs/) This is a list of computer security blogs that all serve a valuable purpose. However,there are thousands of other blogs and people willing to talk about things! So, find what fits you.

If the path is just understanding how to build solid and secure applications then just repeat the above until the end of time. However, if you want to get into the security business there's much, much more to follow!  

### First Project  
The first step to do some hacking or security based project is understanding the technology quite well to start with. So, it's very vital to pick a technology that you consider yourself very familiar with. For the purpose of this, I will pick SQL(structured query language) because of how widely used it is. SQL a language used to store and access information in a unified way. Further knowledge can be found at: [Learning MYSQL](https://www.digitalocean.com/community/tutorials/a-basic-mysql-tutorial) has a good description of how manipulate tables. Further, [Querying and More](https://www.w3schools.com/sql/) teaches basic querying and things. For the purposes of this, and most things you'll ever need to do, just learn how to alter, create and delete tables. Then, understanding what SELECT, FROM, and WHERE are all capable of is enough. After understanding these few things, databases are simple.

#### Step 1
Setting up a database is very, very important in order to use a database! So, this is step 1. It's recommended that this is done on a unix-y system, like Linux or MAC.  
[Setup MySQL](https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-ubuntu-14-04)
:does a great job at getting the database set up locally.

#### Step 2
For the sake of the project, let's use a login screen for this. So, a basic 'table' can be used for this, representing the application. In the table, we will only have two columns:
username and password. This implementation is in Python.
The table will simply look like:

| username | password   |
|----------|------------|
| theGUMan | 1233456789 |
| Spike    | Woof       |  

SQL for this table:
```
CREATE TABLE Login(
    username VARCHAR(50),
    password VARCHAR(200),
    PRIMARY KEY(username)
);
```

SQL Query for getting results:
```
SELECT username, password
FROM Login
WHERE username = 'variable_1' AND
password = 'variable_2';
```
The user needs to give variable_1 and variable_2 to check to see if the authentication works.  

Below is a Python script with how to create a basic login script for the user:

```
def login(username,password):
    #connections
    db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                         user="root",         # your username
                         passwd="passwd",  # your password
                         db="user_fun") #database name

    cur = db.cursor() #crates the connection.
    query = """ #the actual query.
    SELECT *
    FROM Login
    WHERE user = "%s"
    AND password = "%s" ;
    """ % (username,password)

    cur.execute(query) # execute the query

    #Logging information.
    my_string = ""
    for row in cur.fetchall():
        my_string += row[0] + " " + row[1] + "\n"

    if(my_string != ""):
        print "Logged in"
    else:
        print "Failed!"
```

#### Step 3
Here's the fun part: time to just mess around, read around and have fun with it! So, now that the authentication process works, now what? Well, the schema above has some major issues with it...
- **Passwords**:  
Seem funky that the databases are stored by itself in plain text? Well, you were right! What happens in practice is what are called 'one-way hash functions' are used to encrypt the information. A one-way hash function is a way to jumble up data so that it's impossible to go back to the original value. This is used because it can be used to **validate** whether a password is correct, without the database actually storing the password.
- **Privileges**:  
Within databases, as in everything are different types of users, which can do different actions. By having different users with different privileges(ideally as low as possible) it can help with reducing the amount of damage caused by a hacker.
- **SQL Injections**:  
An SQL injection is a way to insert arbitrary code into the system, just with the user input. With this, it's possible to get not only the information initially asked for, but everything from the database! There are thousands of ways to get information out of a database like this! A basic SQL injection for the query `SELECT * FROM Login WHERE username = '%s' AND password = '%s'` where %s is the string being inserted into the query is putting `dummy' OR '1' OR '1`. This will then give the database `SELECT * FROM Login WHERE username = 'dumb' AND password = 'dummy' OR '1' OR '1'`, which will log the user in. Further, we can just use a comment, which is '--' in SQL, to escape the code. This goes on, and on and on! The way to fix this is to use **prepared statements** in the query instead of just plain text. Or, to do an input validation, such as removing all quotations or something like that.

### Step 4
Keep pushing the boundaries of how this authentication process works! Create a [timing based](https://github.com/mdulin2/Timing-Based-SQL-Injection) SQL injection or salt the passwords! The better you understand it, the better you're going to understand how another website might be set up. You'll also understand more about how to make your software more secure!

### What else?
This list is literally endless! There are websites dedicated to helping people learn how certain applications are vulnerable. There are an endless amount of podcasts and blogs that you could listen to, conferences to go to...  
- [Google Grueye](https://google-gruyere.appspot.com/)  
An awfully made web application that is meant to teach people about cross site scripting(XSS) and other issues.
- [Hack this site!](https://hackthissite.org)  
A website with a ton of tutorials and challenges ranging from impossible to beginner.  

### Projects
Personal Projects that can be done:
- [Set up a botnet](https://www.cybrary.it/0p3n/python-programming-hackers-part-6-creating-ssh-botnet/)  
A botnet is a command to control server, that allows you to run commands on someone elses machine.
- [Buillding Your Own website](https://hackernoon.com/build-your-own-react-48edb8ed350d)  
Setting up your own website will open up so many opportunities to build your security skills.

- [Blockchain](https://hackernoon.com/heres-how-i-built-a-private-blockchain-network-and-you-can-too-62ca7db556c0)  
The hip thing right now! It's quite secure; so, understanding how this works would be a blessing!


## Desktop apps

## Video Games

Building video game can be an incredibly fun way to get better at programming and it can also be a great artistic expression.

For the purposes of this section, we'll split up this section into two pieces based on intent. Both paths are respectable goals.

### I want to be a better programmer

If you're building games out of a desire to be a better programmer, this path is for you.

You should build games without a graphical game engine. You're welcome to go down the OpenGL graphics route if you feel comfortable with C++, but this path will focus mainly on non-graphical games libraries.

Note that you will likely make worse games with this path.

#### Pick an Engine

There's typically game engines for every programming language, but here's some good ones for individual languages:

* Python: Pygame
* Java: LibGDX
* Ruby: Ruby2d
* Javascript: Phaserjs

#### A First Project

Start small. Your first project should be moving a block on the screen. Get a window opening. Get a block on the screen. See if you can get the block to move when you press arrow keys.

### I want to make really good games

If you're building games mostly because you want to make games and the programming is more of a means to an ends, this path is for you.

### Unity
//TODO

#### A First Project
//TODO

## Frontend Web

Web is weird. There's a whole lot to learn about it and it can be difficult to understand what to learn when.

This section is primarily about "frontend-only" or "static" sites. That is, they don't connect to a server or database you wrote. The code _only_ runs on the browser.

### First Steps - HTML

The simplest website is just a file called `index.html` with some text in it:

```html
<!-- index.html (note that this is a comment and not actually needed) -->

hi there!
```

You may see other websites with things like `<!DOCTYPE html>` at the top. While important in real projects, they aren't required if you're just learning and you can ignore them for now. The browser will fill them in for you.

Open up your `index.html` file by double clicking on it. Your browser will run it without the need for a server.

If you change your text now, you'll need to refresh your page.

HTML is full of `tags` that label specific content. The most common ones are as follows:

```html
<h1> I make headers! </h1>
<h2> I make headers! </h2>
<h3> I make headers! </h3>
<h4> I make headers! </h4>
<h5> I make headers! </h5>

<p>I make a paragraph with a space afterwards.</p>

<a href="https://github.com">I'm a link!</a>

This is <b>BOLD</b>.
```

There are [a LOT of tags](https://www.w3schools.com/tags/tag_html.asp) out there. Thankfully, they're well documented and there's millions of examples out there! If you see something on a website you want to do, just right click in your browser and "inspect element".

#### Adding style - CSS
TODO


#### Making it do things - Javascript


## Backend Web
//TODO

## Infastructure and AWS
//TODO

## Building Programming Languages

### Overview

Building languages is not as hard as it may appear! In general, the key concepts are:

* [Grammars](https://en.wikibooks.org/wiki/Introduction_to_Programming_Languages/Grammars), or the formal structure of a language.
* Lexing, or the conversion of text into abstract "tokens" that can be used in the grammar. For example, most languages convert the `+` into a `PLUS` token.
* [Parsing](https://en.wikibooks.org/wiki/Introduction_to_Programming_Languages/Parsing), or the conversion of "tokens" (pieces of a language) into an [Abstract Syntax Tree](http://azu.github.io/slide/JSojisan/resources/ast-is-true.png).

### First Project

A good first project is to build a calculator that can compute things like this:

```
1 + 2 * 5
```
https://news.ycombinator.com/
#### Step 1: Write a grammar

You should probably write this in [EBNF form](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form). It doesn't need to have the exact same syntax, but it's useful to understand the conventions.

To get you started, here's a partial grammar for the calculator:

```
add: add '+' add | NUM.
NUM: [0-9].

// TODO: Rest of the grammar
```

Notice that grammars are written as recursive "rules". In the example above, the string `1 + 2` would enter the `add` rule, then recurse till it matched the `NUM`.

#### Step 2: Write a lexer

The easiest way to do this is to use [regular expressions](https://regexr.com/). If you've never used regular expressions before or are generally scared of them, you're also welcome to parse out by iterating through characters.

Don't worry too much about doing it the "right" way at the moment, just focus on getting something working.

You'll typically define your tokens as an `enum` or your language's equivalent.

```javascript
enum Tokens = (
  PLUS,
  NUM
);

function lex(code) {
  let tokens = []

  // TODO: if parse out a '+'
  tokens.push(Tokens.PLUS)

  // TODO: if parse out a number
  tokens.push(Tokens.NUM)

  return tokens
}
```

#### Step 3: Write a parser

The common way to do this is with a [recursive descent parser](http://weblog.jamisbuck.org/2015/7/30/writing-a-simple-recursive-descent-parser.html). This is where your grammar will come in handy.

Your goal here is to create an Abstract Syntax Tree (AST), which is just a tree of your grammar rules!

Each node of the tree should have two values, the type and an optional value.

For example:

```javascript
class Node {
  this.value;    // Any type
  this.type;     // A token or rule name
  this.children; // Other nodes
}
```

When complete, your tree will look a lot like your grammar:

```
# An example input
1 + 2

# An example lexed output
(NUM, 1) PLUS (NUM, 2)

# An example tree
add
  - NUM, 2
  - PLUS, nil
  - NUM, 2
```

The general way people do this is by creating a number of functions corresponding to their rules that "accept" a token and build the part of the tree.

You'll probably want a "look ahead" that checks the next token in the sequence.

```javascript
function add() {
  if (lookahead == Tokens.PLUS) {
    // TODO: add an `add` node to tree and recurse!
    add();
    plus();
    add();
    return;
  }

  // Otherwise, we're a number!
  num();
  return;
}

function num() {
  // TODO: add a num node to tree
  return;
}
```

#### Step 4: Build an interpreter

Now that we have an Abstract Syntax Tree (AST), we can perform a [depth first search](https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/) and evaluate each node.

To start, let's give an example tree:

```
add
  - NUM 2
  - PLUS, nil
  - NUM 3
```

We typically build our interpreter similar to how we built our recursive descent parser; we build it with functions. Except this time, our functions can visit each child node and return a value.

For example:

```javascript
function visit(node) {
  // TODO: search down the tree until we get to a num and then return
  // that value
}

function visitAdd(node) {
  return visit(node.add[0]) + visit(node.add[0]);
}

function visitNum(node) {
  return node.value;
}
```

### Where to go from here

The world of programming languages is really, really big.

#### Compiler Compilers!

If you went through and built that recursive descent parser and found it to be a little tedious, you're in luck! There's options for auto-generating a recursive descent parser from a grammar.

* [Antlr](http://www.antlr.org/) is my personal favorite!
* [Yacc, Lex, and Bison](http://dinosaur.compilertools.net/)

#### Formal grammars

Many bugs you've faced can come down to ambiguities in your grammar. Having a better understanding of the different types of grammars and parsers and how they're defined can really help you.

* [LL(n)](https://en.wikipedia.org/wiki/LL_parser)
* [LALR](https://en.wikipedia.org/wiki/LALR_parser)
* [PEG](https://en.wikipedia.org/wiki/Parsing_expression_grammar)

### Example projects

#### A Portfolio Language

Build a language like markdown that creates portfolio websites for you!

#### Code Generators

Create an AST generator for an existing language and use that to generate code for the language.

#### AST Formatter

Create a formatter that takes an existing programming language, converts it to an AST, then reprints it the correct style.

#### Build a simple scripting language

Build another language in your favorite language! Add variables, functions, arrays, and more!

## Learning specific programming languages

### Python
Python is a beautiful language to pick up. It's simple syntax, easy to read and great for writing quick scripts. Sometimes the code is so beautiful, it's referred to as 'Pythonic Beauty'. The language is also very flexible. To make things better, it has an unbelievable amount of libraries made already for us to use! However, it can be slow some of the time, sadly.
- [10 Reason to Use Python](https://medium.com/@joshdotai/10-reasons-python-is-awesome-3dcb98a1291a)

#### First Program
Of course, the first program has to be print hello world.   
Python makes this quite simple; lucky for us!  
```
print("hello world")
```
Boom, your first Python program!  

#### Basic Python  
In Java, C++ and quite a few other languages, scope is defined with curly braces ({). A function may look like
`int get_val(int x){ return x;}`.  
Instead of using curly braces to show scope, Python uses tabs. The code above in C++ is identical to
```
def get_val(x):
    return x
```
Notice the tab above to represent the scope to return x.
Another thing you have probably realized by now is that Python also doesn't define types. This is referred to as *type inference*. In C++, to define a variable we must use `int x = 5;`. But, in Python, the code is just `x = 5`. The language is able to figure out that the variable x is of type int simply because the right hand side, 5, is an integer. Type inference can make Python very nice to code in, but difficult to decode at times. It's easy to be comparing an int to a string, and take forever to find it.

#### Basic Constructs

- Types:  
int, float, boolean, string, list, dict, are the basic types that are great to know. int, float and string are self explanatory. A boolean value is a true or false, a binary representation of something. Then a list is essentially an array in C++. Dictionary is a key, mapped to a value of some kind.

- Operators:  
Plus(+), minus(-), multiple(\*), divide(/), modulo(%). The first three are quite simple to use, with no strange perks. However, be careful with the **division** operator. For instance, 5/2, will not be 2.5 but 2. This is done because of integer division. So, be very careful when using division to make sure the right type of value is being used. By adding a float type definition around the divisor, this will fix this issue. Lastly, modulo takes the remainder of the division done.

- Conditional Operators:  
The operators are ==, >=, <=, !=, <, >. These are self explanatory so it's not needed to talk much about much more. != is the not equal operator.

- Lists:  
Lists are a great way to store data! Lists are a ordered list, which are really nice to deal with. If we have the list of numbers 1,2,3,4,5, then we write `lst = [1,2,3,4,5]`. Accessing the first element is `lst[0]`. To access the last item in the list use a -1 as the index.  

- For loops:  
The below will display 1,2,3,4,5
```
for index in range(5):
    print(index)
```
To iterate over a list, it's like this:
```
for element in lst:
    print element
```
With what is above, the element represents the nth item in the lst.

- Functions:  
A function is a way to split up code such that it makes the program smaller, reusable and easier to follow. In Python, a function is defined with the `def function_name(par1, par2):` An example of a function that adds 10 to a value then divides by 2 is:  
```
def add10_div2(x):
    val = (10+x) /2
    return val
```

- If statements:    
In order to do conditional statements in Python,, if statements are used. The basic constructs are if, elif and else. if is the initial if statement, elif is the conditional statements following and else is the final clause.  
```
if(x == 1):
    return 1
elif(x >= 2):
    return 2
else:
    return 3
```

#### Using Python!
Let's do something a little more complicated! Fizzbuzz sounds like a great choice!
```
# single line comments
def Fizzbuzz(amount):
    """
    multi-line comments
    """
    for i in range(amount):
        if(i % 3 == 0):
            print("fizz")
        elif(i % 5 == 0):
            print "buzz"
        elif(i % 15 ==  0):
            print("Fizzbuzz!!!")
        else:
            pass
    return 1
```

#### Deeper  
These are the absolute bare bones of Python; so, there is so much more to learn!  
Here are a few resources:  
- [The New Boston with Bucky Roberts](https://www.youtube.com/watch?v=hnxIRVZ0EyU):  
Is one of the best in the business for learning programming languages. Highly recommended!  
- [Py Slackers](https://pyslackers.com/):  
A group of people who love all things Python!  
- [w3schools](https://www.w3schools.in/python-tutorial/):  
Another step by step tutorial

#### Miscellaneous
A few more things about Python should be stated:  
- Pip:  
Pip is the Python package installer used on most systems. The command looks like `pip install numpy`, where numpy is the package.  
- Where to code with it:  
Using a text editor then compiling in the terminal is a common option for this. A few text editors that work well are atom, visual studio code and sublime. This will give a person more control over the environment themselves, but can be more difficult to maintain. Now, there is also an **IDE(Integrated Development Environment**) where the code is written in the text editor then compiled inside of this text editor. A few Python IDE's are PyCharm, PyDev and IDLE.











### Ruby

//TODO

### C++

//TODO

## Functional Programming

//TODO

