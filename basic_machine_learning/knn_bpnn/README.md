# knncross-bpnn

http://web.eecs.utk.edu/~qi/ece471-571/project/proj3.htm

## Task 1 (30): kNN Improvements.
* Task 1.1: Use the "fglass" data set. Experimenting cross-validation with 
    kNN as the classifier. 10 subsets are provided for the dataset. Use 10-fold 
    cross validation. Try out different k's. Based on the performance from cross-validation, 
    determine the best k for this data set.
* Task 1.2: For the best k you found from Task 1.1, use Euclidean and other Minkowski 
    distance of different degrees, and report their performance. (Cross-validation is assumed)
    
### data.txt
The data comes from Pattern Recognition and Neural Networks by B.D. Ripley 
(http://www.stats.ox.ac.uk/pub/PRNN/)

The data column names are:
RI     Na   Mg   Al   Si     K   Ca   Ba   Fe type

Each row is a data sample

### data_index.txt
There are 10 rows for 10 fold cross validation. Each row lists the row indices in data.txt
for the testing samples in that fold. 

### data.py
Assists in loading in the data

### norm.py
Normalizes the data

### knn.py
Classifies the data using knn for a given k and minkowski distance

### knncross.py
Runs two knn experiments using cross validation and plots the results
* Find the most accurate k based on a minkowski distance of 2
* Find the most accurate minkowski distance based on the most accurate k
    
## Task 2 (50): BPNN Implementation. 
In Homework 3, you have experienced that Perceptron cannot learn an XOR logic.
* (for 471 students): Use the MATLAB package or any open-source package and implement BP to learn the XOR logic.
* (for 571 students): Implement 3-layer BPNN in C/C++ or Python or MATLAB (non-EECS) yourself. 471 students will get 10 bonus points if choosing to implement yourself.
 
Experimenting effects of: 
* different initialization
    * 0 should not work 
    * Chosen constant initializations (-1 to 1 by 0.1, -0.1 to 0.1 by 0.01)
    * Random uniform distribution
* different activation functions
    * S(x) = signum(x) = 1 if x>=0, -1 if x < 0
    * S(x) = sigmoid(x) = 1 / (1 + e^{-x})
    * S(x) = improved_sigmoid(x) = (2a / (1 + e^{-bx})) - a, a = 1.716, b = 0.666666
* different learning rate
    * 0.1 is "often adequate"
    * 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003
* the use of momentum (latex legible)
    * w_{st}^{k+1} = w_{st}^k - c^k * (dE^k)/(dw_{st}^k)
    * w_{st}^{k+1} = w_{st}^k + (1 - c^k) * delta(w_{bp}^k) + c^k(w_{st}^k - w_{st}^{k-1})

### Notes
bpnn was implemented as explained in this post https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
Here are some maybe helpful notes as to why/how I implemented things. A 2 node input layer, 2 node hidden layer, 1 node output layer
was assumed, but I figured if I assumed variable structure it would help me later and help me to make better informed
choices regarding the data setup:
* keep a vector of the number of nodes per layer
* Node Value Representation
    * index 1: layer the value is in
    * index 2: node the value is in
* Weight Representation
    * index 1: layer the weight is coming from 
    * index 2: node the weight is coming from
    * index 3: node the weight going to (in next layer)




   
