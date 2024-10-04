> This is the second part of Siyuan's note on Andrew Ng's "Advanced Learning Algorithms" course.

Let's get started.

# 8 Activation Functions

## 8.1 Alternatives to the sigmoid activation

> We were using sigmoid activation all the time because we were building up neural network by creating logistic regression units and string them together.
>
> But, if you use other activation functions, your neural network can become **much more powerful**.

Recall the **demand prediction example** we were using last week:

> The content of this example: given price, shipping cost, marketing material **>>>** try to predict if something is highly affordable, if there's good awareness and high perceive quality and based on that **>>>** try to predict if it's a top seller.

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/06/17045322320136.jpg)

Here is the **problem** this neural network is facing:

- It seems like the degree to which possible buyers are **aware** of the T-shirt, may **not** be **binary** (can be “little bit aware”, “somewhat aware”, “extremely aware” or could have gone “completely viral”). 
- So, maybe it will be more appropriate that “**awareness**” can be any **non-negative number** (0 to infinite).

previously, we used this function to calculate the activation of that second hidden unit estimating awareness:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/06/17045331513973.jpg)

- $g()$ was a sigmoid function, its output goes between 0 and 1.

Now, if we want $\vec a^{[1]}_2$ to potentially take on much larger positive values, we should swap in a different activation function:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/06/17045331514056.jpg)

About this function:

- if $z < 0$, then $g(z) = 0$ (left half of this diagram).
- and, if $z ≥ 0$, then $g(z) = z$ (right half of this diagram).
- and the mathematical equation for this is: $g(z) = max(0,z)$.

This is, the **ReLU** (Rectified Linear Unit).

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/06/17045416195777.jpg)

More generally, you have a choice of what to use for $g(z)$, the activation function. Here are the most commonly used activation functions:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/06/17045417435618.jpg)

- The “Linear activation function” (left), $g(z)$ is just equals to $z$.

  > sometimes people would say “we are not using any activation function”, because if $g(z) = z$, there is no “$g$” at all.

- So, $a^{[1]}_2$ here is just $\vec w^{[1]}_2 \cdot \vec x+b^{[1]}_2$ .

Later this week, beside these three activation functions, we'll touch on the fourth one called the **Softmax** activation function. With these activation functions, you'd be able to build a rich variety of powerful neural networks.

> But, how do you choose between these different activation functions? Let's take a look at that in the next video.

## 8.2 Choosing activation functions

> Let's take a look at how you can choose the **activation function** for different neurons in your neural network.

We'll start with some guidance for how to choose it for the **output layer**:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/07/17046152547858.jpg)

When considering the activation function $g(z)$ for the output layer, it turns out that there will often be one fairly natural choice depending on what is the targets of the ground truth label $y$.

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/07/17046153300291.jpg)

1. for example: classification problem. $y$ is either $0$ or $1$, so binary classification problem, so, **sigmoid function** will be naturally considered.

2. Alternatively: if you’re trying to predict how tomorrow’s **stock price** will change compared to today’s stock price, then, $y$ can either be positive of negative number, then, use **linear regression** instead.

   > because linear regression model $g(z_1^{[3]})$ can take either positive or negative values.

3. if $y$ can only take on **non-negative** values, such as if you’re predicting the **price of a house**, then the most natural choice would be the **ReLU** activation function.

   > because this activation function only take non-negative values.

So, when choosing the activation function to use for your output layer, usually depending on what is the label $y$ you're trying to predict, there'll be one fairly natural choice.

How about the hidden layers?

> The **ReLU** activation function.

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/07/17046168289609.jpg)

- the **ReLU** activation function is by far the **most common choice** in how neural networks are trained by many practitioners today.

- Sigmoid function was once the most popular (in early history of ML). Now we hardly use it.

  > Now, the only reason you use the sigmoid function in output layer is that you’re solving a **binary classification** problem.

- A few reasons people don’t use sigmoid activation today:

  1. **Speed (faster calculating)**. The ReLU is a little bit faster (the function itself more simpler).

  2. **Shape (faster learning)** (more important). The ReLU only goes flat in one part of the graph (second quadrant), which makes gradient descent algorithm achieving the goal **much more faster**. 

     Why?

     > when $g(z)$ is flat, then $\frac {\partial} {\partial w} J(W, B)$ will be approximately $0$, then with small gradients, it slows down learning.

     ![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/07/17046166214133.jpg)

- As a result, the **ReLU** activation function has become now by far the most common choice.

So, that’s how we choose activation functions for hidden layers as well.

To summarize how we choose the activation functions:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/07/17046173110192.jpg)

- **For the output layer**:

  - **Sigmoid** if you have a binary classification problem
  - **Linear** if $y$ is a number that can take on positive or negative values
  - **ReLU** if $y$ can take on non-negative values.

- **For the hidden layers**: use ReLU as a default.

  in TensorFlow, this is how you would implement ReLU:

  ![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/07/17046172688889.jpg)

With these richer set of activation functions, we be well positioned to build much more powerful neural networks than just once using only the sigmoid activation function.

> Every few years researchers sometimes come up with another interesting activation function and sometimes they do work a little bit better. But I think for the most part and for the vast majority of applications what you learned about in this video would be **good enough**.

With that, I hope you also enjoy practicing these ideas, these activation functions in the **optional labs** and in the **practice labs**. 

In the next video, let's take a look at why do we even need activation functions, and **why activation functions are so important** for getting your new networks to work.

## 8.3 Why do we need activation functions?

If we were to use the linear activation function in every neuron in the neuron network, recall this demand prediction example:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/08/17046723721580.jpg)

It turns out that: this big neural network will become **no different than just linear regression**. 

- This would defeat the entire purpose of using a neural network
- because this model would then just not be able to fit anything more complex than the linear regression model

> my understanding: the mixed variety of activation functions together apply into one neural network, would make this neural network more “**complex**” >>> being more able to fit this true world with complexity.

Let's illustrate this with a simpler example: 

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/08/17046733455843.jpg)

- only 1 feature, input $x$, is just a number
- 1 hidden layer with 1 neuron, with parameters $w_1^{[1]}$ and $b_1^{[1]}$
- hidden layers outputs $a^{[1]}$, which is just a number
- 1 output unit, with parameters $w_1^{[2]}$ and $b_1^{[2]}$
- outputs $a^{[2]}$, just a number.

Let’s see what this neural network will do if we were to use the linear activation function $g(z)$ everywhere:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/08/17046737919854.jpg)

- first calculate $a^{[1]}$, then $a^{[2]}$, then substitute $a^{[1]}$ into $a^{[2]}$ >>> finally we discover that $a^{[2]}$ is just another $wx + b$.

  > So, we might have just use a linear regression model at first..

- “if you're familiar with linear algebra, this result comes from the fact that a linear function of a linear function is itself a linear function.” And this is why having multiple layers in a neural network **doesn’t let the neural network compute any more complex features or learn anything more complex and just a linear function**.

So in the general case:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240108103344.png"/>

- If you use linear regression in every unit, then the output $\vec a^{[out]}$ can be expressed as a linear function of the input feature $x$.  

- Alternatively, if we were to still use a linear activation function for all the hidden layers, but we were to use a logistic activation function for the output layer, then it turns out that this model becomes **equivalent** to **logistic regression**.

  $\vec a^{[4]}$ can be expressed as:

  $$
  \vec a^{[4]} = \frac {1} {1+e^{-(\vec w_1^{[4]}\cdot\vec a^{[3]}+b_1^{[4]})}}
  $$

  > so this big neural network doesn't do anything that you can't also do with **logistic regression**.

So, don't use the linear activation function in the hidden layers of your neural network, ReLU activation function should do just fine.

Next:

> In the next video, I'd like to share with you a generalization of what you've seen so far for classification in particular when y doesn't just take on two values, but may take on 3 or 4 or 10 or even more categorical values.

# 9 Multiclass Classification

## 9.1 Multiclass

Multiclass classification refers to classification problems where you can have **more than just two possible output labels**.

Example: 

1. If you're trying to read postcodes or zip codes on an envelope, there are actually 10 possible digits you might want to recognize. **That's multiclass classification problem**.
2. if you're trying to classify whether patients may have any of three or five different possible diseases, **that's multiclass classification problem**.
3. When you  look at the picture of a pill that a pharmaceutical company is manufactured and try to figure out does it have a scratch defect or discoloration defect or a chip defect, **that's multiclass classification problem**.

So, a multiclass classification problem is still a classification problem in that $y$ can take on only a small number of discrete categories (more than 2, but not every number).

This is what it looks like:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240108105207.png"/>

- The model would compute the possibilities that $y$ is equal to 1, 2, 3 or 4 (four different categories).
- The algorithm we learn about in the next video can learn a decision boundary that maybe looks like this (on the right part) that divides the space X1 and X2 into four categories.

Next: 

> We'll look at the **Softmax regression algorithm**, which is a generalization of the logistic regression algorithm, to carry out multiclass classification problems.

## 9.2 Softmax

Let's take a look at how it works!

First, recall the logistic regression:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240108105944.png"/>

- we use $g(z)$ to compute the possibilities that $y = 1$.
- if $P(y=1|x) = 0.71$, then $P(y=0|x) = 0.29$.

Then, let's now generalize this to Softmax regression:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240108111809.png"/>

How to get the Softmax regression function (left half of this slide):

1. compute $z_1$, $z_2$, $z_3$ and $z_4$ respectively. These are the parameters of the function.

2. for each possible value of y, based on $z$, compute $a_1$, $a_2$, $a_3$ and $a_4$ (**Note**: $a_1 + a_2 + ...+a_n = 1$). 

   > These $a_1$, $a_2$, $a_3$ and $a_4$ computes possibilities that $y=1$, $y=2$, $y=3$ and $y=4$. For example, 0.30, 0.20, 0.15, and 0.35.

Another interesting fact: 

> If you apply Softmax regression with $n = 2$, so there are only two possible output classes, then Softmax regression ends up reducing to a **logistic regression** (even though the parameters will be a little different)

Then, let's specify the **cost function** for Softmax regression:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240108114532.png"/>

- Previously the logistic regression: 

  - define $a_1$ and $a_2 = 1-a_1$, then loss function is clear
  - specifically, if $y=1$, the loss function would be $- \log a_1$
  - otherwise, if $y=0$, then the loss function would be $- \log a_2$
  - Then the cost function is average loss.

- Then, the **Softmax regression**:

  - notice the $loss(a_1,...,a_n,y)$, the similarity with the logistic regression.

  - To illustrate this function, we can visualize the crossentropy loss: 

    the graph for the function:
    $$
    L = -\log {a_j}
    $$
    is this:

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240108114844.png"/>

    We discover: **the smaller $a_j$ is, the bigger the loss**. This incentivizes the algorithm to make $a_j$ (which is, the possibility that $y=j$) **as close to 1 as possible**.

  - We need to only compute one loss function for one training example. For example, if $y=2$, you end up computing $-\log{a_2}$, only this (same as logistic regression's loss function).

- My question here: so, what the gradien descent of this "cost function" of Softmax regression is actually doing? 

  - Reducing the average loss. Then? How does this work... 

  - To clarify what the "loss" mean... it measures how well you're doing on **one training example**. 

    In logistic regression, by judging the predicted $\hat y$, if accurate ($\hat y$ close to 1) , loss ($- \log \hat y$, also $- \log {f_{\vec{w},b}(\vec{x}^{(i)})}$) is little, otherwise large. 

    - if $y=0$, then if $- \log(1- {f_{\vec{w},b}(\vec{x}^{(i)})})$ close to 0, loss is little. Otherwise large.

    In linear regression, this is loss (squared error): $(\hat{y}^{(i)}-y^{(i)})^2$ 
  
  - So here, $a_1,...,a_n$ is the predicted values of possibility of $y$. by minimize the average loss (gradient descent), by
  
    1. for each training example (ground truth label $y$ is one of these possible outputs), calculate the possibilities for each of these possible outputs
    2. compute $loss = - \log a_j$ ($j$ is corresponded to $y$). If this possibility is far away from $1$, then  $loss$ will be huge, then the parameters of softmax should be optimized in a larger step.
  
  - this function reduces error between possiblility and reality for each training example, and each possible output value  (by optimizing the parameter $z$ >>> optimizing the $w$ and $b$), then this learning happens. Good.
  

Next: 

> take this Softmax regression model, and **fit it into a neural network**, to train a neural network for multi cost classification.

## 9.3 Neural Network with Softmax output

> In order to build a neural network that can carry out multiclass classification, we're going to take a Softmax Regression unit and put it into the **output layer**.

Previously in the neural network designed for handwritten digit recognition, we use one output unit:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109103345.png"/>

Now, in order to classify 10 different numbers, we need to change this neural network to have **10 output units** like so:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109103502.png"/>

And this new output layer is a **Softmax** layer. When doing forward prop in this neural network, firstly use relu layer to compute $\vec a^{[1]}$, then $\vec a^{[2]}$, finally the $\vec a^{[2]}$ is input in Softmax layer (**Softmax converts a vector of values to a probability distribution**), and output $\vec a{[3]}$, the final choice of the number.

Then let's delve into what happens in this output layer:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240119113603.png"/>

- firstly, compute $z_1^{[3]}$, $z_2^{[3]}$ through $z_{10}^{[3]}$

- then, $a_1^{[3]}, a_2^{[3]},...,a_{10}^{[3]}$ 

  > $a_1^{[3]}$ output the chance that $y=1$ 
  >
  > ... 
  >
  >  $a_{10}^{[3]}$, outputs the chance that $y=10$.

- One property that is **unique** for the Softmax function: each of these activation values depends on **all of the values of $z$** (not just the corresponding $z$ itself)

  State it differently:

  > If you want to compute $a_1$ through $a_{10}$, that is a function of $z_1$ all the way up to $z_{10}$ simultaneously.

Finally, let's look at how you would implement this in TensorFlow:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109111238.png"/>

Still, 3 steps to train a model:

1. **Step 1**, to tell tensorflow to string these layers (requires **numbers of units** in each layer and the **activation function** of each layer.)

2. **Step 2**, tell TensorFlow to compile the model with **loss function** specified. 

   here, the loss function is "**Sparse Categorical Cross entropy function**"

   - *Sparse categorical* refers to you need to classify $y$ into categories. 
   - *Sparse* refers to, $y$ can only take on one of these 10 values (each digit is only one of these categories).

3. **Step 3**, the code for training the model is the same as before.

**Kind reminder**: this given code will work, but **don't actually use this code,** because there will be a better version of the code that makes TensorFlow work better (later in this week, next video).

Next:

> There's a different version of the code that would make TensorFlow able to compute these probabilities much more accurately. Let's take a look at that in the next video.

## 9.4 Inproved implementation of softmax

A better way to implement this algorithm in TensorFlow.

Start with a little test, 2 different ways of computing the same quantity in a computer:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109111610.png"/>

Let's illustrate this in this notebook:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109112008.png"/>

- It turns out that, while the way we have been computing the cost function for Softmax is correct, there's a different way of formulating it that **reduces these numerical round of errors** leading to more accurate computations within TensorFlow.

Why? First explain this a little bit more detail using **logistic regression**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109113514.png"/>

- It turns out that, if you allow TensorFlow to not have to compute $a$ as an intermediate term, but instead, if you tell TensorFlow that the loss is this expression down here (take $a$ and expand it into its expression):
  $$
  loss = -y\log{(\frac{1}{1+e^{-z}})-(1-y)\log{(1-\frac{1}{1+e^{-z}}})}
  $$
  this is a more numerically accurate way to compute this loss function.

- Whereas the original procedure was like to insisting on compute $(1+\frac {1}{10000})$ and $(1-\frac {1}{10000})$ , then manipulating these two to get $\frac {2}{10000}$ .

- But instead by specifying this expression directly, it gives TensorFlow more **flexibility** in terms of how to compute this and whether or not it wants to compute it explicitly (not insist on the intermidiate value $a$ anymore).

- The optimized code is here:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109113736.png"/>

  Changes:

  1. Change the activation function of output layer to the **linear activation function**.
  2. Put both the activation function $\frac{1}{1+e^{-z}}$ as well as this `BinaryCrossentropy` loss into the specification of the loss function (that's what `from_logits = True` argument ask TensorFlow todo).

  Explain: What "logits" are?

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109114304.png"/>

  - "logit" is basically the "$z$".
  - So, TensorFlow will compute $z$ as a intermediate value, but they can rearrange terms to make this become computed **more accurately**.

Now let's take this idea and apply to Softmax regression:



- First, recall the activation and loss function is this:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109114617.png"/>

- And that's the original code we have seen before (right part):

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109114921.png"/>

- Again, if you **specify what the loss** is (get rid of the intermidiate value, expand the expression of $a$): 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109114823.png"/>

  Then, this gives TensorFlow the ability to rearrange terms and compute this in a **more numerically accurate** way.

- The updated code is shown here:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109115215.png"/>

  the same change as before!

  1. Change the activation function of output layer to the **linear activation function**.
  2. Add `from_logits = True` argument.

  This version of code is recommended is **more numerically accurate** although unfortunately it is a little bit **harder to read** as well.

And conceptually this code does **the same thing as the first version that you had previously**, except that it is a little bit more numerically accurate.

Now, there's just one more detail:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109115721.png"/>

- Now, we use a linear activation function routed in a Softmax activation function. So the neural network's final layer **no longer output these probabilities** $a_1,...a_{10}$, insted it outputs **logits** $z_1,...z_{10}$.

And, we didn't yet talk about this in the case for **logitic regression**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240109115648.png"/>

You also have to change the code this way 👆 to take the output value, and map it through the logistic function in order to actually get the probability. 

- Note: I want to specify the difference between now and old code..

Next:

> Before wrapping up multiclass classification, I want to share you one other type of classification problem called a **multi-label classification problem**.

## 9.5 Classification with multiple outputs

There's a different type of classification problem called a **multi-label classification problem**. 

Here's an example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110105817.png"/>

- The input $\vec x$ are three different **labels** corresponding to whether or not there any **cars, buses or pedestrians** in the image.

- So, the output $\vec y$ is actually **a vector of 3 numbers**, corresponding to the presence of each kind of item in this image.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110110238.png"/>

  compared with ""hadwritten digit classification" example before, the previous example, the output $y$ is only **one number**, not a vector.

How do you build a neural network for multi label classification?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110111316.png"/>

- One way to go about it is: Just treat this as 3 **completely separate machine learning problems**. Build three neural networks:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110110821.png"/>

  - One neural network to decide are there any cars

  - The second one to detect buses 
  - The third one to detect pedestrians

- Another way:  to train a **single** neural network to **simultaneously detect all three** of cars, buses and pedestrians. Your neural network architecture looks like this:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110111335.png"/>

  Pay attention to the final output layer, which now contains **3 output units**, and outputs a vector $\vec a^{[3]}$.

  - use sigmoid activation function for each of these nodes in the output layer.

So, **multiclass** classification and **multi-label** classification are sometimes confuse with each other. Be clear with the definiton of each one, and choose the right one depending on the practical needs.

That wraps up the section on multiclass and multi-label classification.

Next:

> We'll start to look at some more advanced new network concepts, including an optimization algorithm that is even better than gradient descent. It'll help you to get your learning algorithm to learn much faster.

# 10 Additional Neural Network Concepts

## 10.1 Advanced Optimization

There are now some other optimization algorithms for minimizing the cost function, that are even better than gradient descent.

This is the expression for one step of gradient descent:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110112954.png"/>

How can we make this work even better? First look at how it optimizes the parameters:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110113330.png"/>

- In a contour plot, step by step to the "lowest point".
- The speed of step is determined by learning rate $\alpha$.

Why don't we make $\alpha$ bigger?

Can we have an algorithm to automatically increase $\alpha$? >>> Adam algorithm.

Then, look at this plot (on the right):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110113727.png"/>

- $\alpha$ is too big.

Adam algorithm can also automatically **decrease** the value of $\alpha$. Let's see what it is.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110114356.png"/>

- "Adam" stands for: **Adaptive Moment Estimation**

- the Adam algorithm doesn't use a single global learning rate $\alpha$. It uses a **different learning rates** for **every single parameter** of your model.

  If you have parameters $w_1$ through $w_{10}$, as well as $b$ , then it creates 11 learning rate parameters $\alpha _1$ to $\alpha _{11}$ 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110114050.png"/>

The intuition behind the Adam algorithm is:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110114859.png"/>

- If parameter $w_j$, or $b$ seems to keep on moving in roughly the **same direction**, then **increase** the learning rate for that parameter (Le's go faster in that direction!).
- Conversely, if a parameter keeps **oscillating back and forth** (the second example), then reduce $\alpha_j$ for that parameter a little bit

> The details of how Adam does this are a bit complicated and **beyond the scope of this course**, but if you take some more advanced deep learning courses later, you may learn more about the details of this Adam algorithm.

But in codes this is how you implement it:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240110114834.png"/>

- **Step 1**, The model is exactly the same as before

- **Step 2**, the way that you compile the model is similar as before, but now we need to specify the optimizer by the `optimizer` parameter. Use this:

  ```python
  tf.keras.optimizers.Adam(learning_rate=1e-3)
  ```

   And the `learning_rate=le-3` here is the default learning rate that needed, $10^{-3}$.

  > When you're using the Adam algorithm in practice, it's worth **trying a few values for this default global learning rate**. Try some large and some smaller values to see what gives you the fastest learning performance.

That's it for the Adam optimization algorithm. It typically works much faster than gradient descent, and it's become a defacto standard in how practitioners train their neural networks.

Next: 

> To touch on some more advanced concepts for neural networks, and in particular, in the next video, let's take a look at some **alternative layer types**.

## 10.2 Additional Layer Types

> It turns out that just using the **Dense** layer type, you can actually build some pretty powerful learning algorithms. And to help you build further intuition about what neural networks can do. 
>
> However, there's some **other types of layers** as well with other properties.

To recap in the **Dense** layer:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240111104913.png"/>

The peoperty of dense layer:

- The activation of a neuron, is a function of every single activation value from the **previous layer**.

One other layer type that you may see in some work is called a **Convolutional layer**:

> Let's start by using a hand-written digit recognition example (like write a 9 in an image):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240111105850.png"/>

The feature for this kind of layer: 

- Each neuron is not going to look at the entire input image $\vec x$. instead, it's only going to look at the pixels in a **limited region** of the image.

- So, why might you want to do this? Some reasons:

  1. Speeds up computation

  2. Need less training data (alternatively, it can also be less prone to overfitting)

     > next week, we will dive into greater detail on "overfitting", when talk about pratical tips.

- If you have multiple convolutional layers in a neural network, then sometimes that's called a **convolutional neural network**.

Illustrate more details on the convolutional layer:

> In this example, we use **1D input**, rather than 2D input.

So, if you put two electrodes on your chest you will record the voltages that look like this, corresponds to your heartbeat:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240111112956.png"/>

We can try to train a neural network on EKG signals to try to **diagnose** whether a patient has a **heart issue**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240111112916.png"/>

- EKG signals is just **a list of numbers**, corresponding to the height of the surface at different points in time. 

- For example, we may have **100 numbers** $x_1,x_2,...,x_{100}$corresponding to the height of this curve at 100 different points of time.

- Here is how we construt our neural network. Initially, the input feature $\vec x$ is in, then to activate the **first hidden layer**:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240111112839.png"/>

  - Let **first hidden unit** to look at $x_1,...,x_{20}$
  - **Second** hidden unit, to look at $x_{11},...,x_{30}$ 
  - **Third** hidden unit, $x_{21},...,x_{40}$
  - ...
  - **Final** hidden unit, $x_{81},...,x_{100}$

  So, that's 9 units in this first hidden layer.

- The next layer can also be a convolutional layer (each unit now, not to look at whole 9 units from the previous layer):

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240111112722.png"/>

  - **First hidden unit**, look just **first 5 activations** from the previous layer, $a_1^{[1]},a_2^{[1]} ...a_5^{[1]}$
  - **Second hidden unit**, just another five numbers, $a_3^{[1]},a_4^{[1]} ...a_7^{[1]}$
  - **Third hidden unit**, $a_5^{[1]},a_6^{[1]} ...a_9^{[1]}$

- Finally the output layer: a **sigmoid unit** that look at all 3 activation numbers from the previous layer.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240111112511.png"/>

  > This layer make a binary classification regarding the **presence or absence of heart disease**.

With convolutional layers, you have many architecture choices, such as:

- **How big is the window** of inputs that a single neuron should look at.
- **How many neurons** should layer have. 

By choosing those architectural parameters effectively, you can build new versions of neural networks that can be even more effective than the dense layer for some applications.

> I'm not going to go deeper into convolutional networks in this class. But I hope that you find this additional intuition that neural networks can have other types of layers as well to be useful.

Some expansions on the cutting-edge researchs:

> if you sometimes hear about the latest cutting edge architectures like a transformer model or an LSTM or an attention model.
>
> A lot of this research in neural networks even today pertains to researchers trying to invent new types of layers for neural networks. And plugging these different types of layers together as building blocks to form even more complex and hopefully more powerful neural networks.

Next:

> we'll start to talk about practical advice for how you can build machine learning systems.

# 11 Backprop Intuition

## 11.1 What is a derivative?

> TensorFlow will automatically use back propagation to compute derivatives and use gradient descent or Adam to train the parameters of a neural network.

 The **back propagation** algorithm, which computes derivatives of your cost function with respect to the parameters, is a key algorithm in neural network learning. 

**But how does it actually work?** In this and in the next few optional videos, we'll try to take a look at *how back-propagation computes derivatives*.

Example: use a simplified cost function (ignore $b$ for this example).

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116164721.png"/>

- When $w=3$, if value of $w$ goes up $0.001$, then $J(w)$ roughly goes up $0.006$ (6 time of $w$).

  > if $\epsilon$ where **infinitesimally** small (very very small), then the change of $J(w)$ will be closer to "6 times $w$".

- In calculus, the derivative of $\frac{\partial}{\partial w} J(w)$  is **6**.

This leads to an **informal definition** of derivative:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116165519.png"/>

- The definition is: 

  > whenever $w$ goes up by a tiny amount $\epsilon$, that causes $J(w)$ to go up by $k \times \epsilon$, then we say that **the derivative of $J(w)$ is equal to $k$**

- This is why when implementing gradient descent, we repeatedly use this rule👇 to update the parameter $w_j$ 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116165403.png"/>

A few more examples of derivatives:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116170348.png"/>

- we can see in $J(w)=w^2$:

  - when $w=3$, then $\frac{\partial}{\partial w} J(w) = 6$
  - when $w=2$, then $\frac{\partial}{\partial w} J(w) = 4$
  - when $w=-3$, then $\frac{\partial}{\partial w} J(w) = -6$

- Then, we plot this $J(w)$:

  the derivatives corresponds to the slope of a line that just touches the function $J(w)$ at a specific point.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116173845.png"/>

  - Slope equal to 6 when $w=3$ 
  - Slope equal to 4 when $w=2$ 

- So, the calculus allow us tocalculate the derivative of $J(w)$ in respect to $w$ as $2w$  >>> 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116174215.png"/>

  next, we'll use Python to compute this, based on the library called "**SymPy**".

More examples:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116174519.png"/>

- How can we calculate these new derivatives when $J(w)$ is $w^3$, $w$ or $\frac {1}{w}$ ? Use Python with **SymPy** lib:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116175733.png"/>

  1. tell **SymPy** we're goin to use `J` and `w` as symbols for computing derivatives.

     ```python
     J, w = sympy.symbols('J,w')
     ```

  2. Then tell SymPy our cost function `J` is `w` squred.

     ```python
     J = w**2
     ```

     We can see what SymPy output is in a nifty font (automatically rendered).

  3. Use SymPy to take the derivative of `J` with respect to `w` , then it tells us it's $2w$

     ```python
     dJ_dw = dsympy.diff(J,w)
     ```

  4. We can plug in some actual value of `w` to evaluate the derivative:

     ```python
     dJ_dw.subs([(w,2)])
     #output: 4
     ```

  5. Try `J` is  `w` cubed, and some more `J` expression:

     ```python
     J = w**3 #cubed
     ```

     <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116175829.png"/>

     ```python
     J = w #w itself
     ```

     <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116175923.png"/>

     ```python
     J = 1/w #1 over w
     ```

     <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116175622.png"/>

- To conclude:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116180245.png"/>

  This is derivative.

One last thing, a note on derivative notation:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240116181423.png"/>

For historical reasons, calculus text book will use these two different notations depending on whether $J$ is a function of a **single variable** or a function of **multiple variables**:

- if $J(w)$ is a function of a **single variable** $w$, then derivative is written in this form $\frac {d}{dw}J(w)$
- If $J(w_1,w_2,...,w_n)$ is a function of **more than one variable**, then we use $\frac {\partial}{\partial w_i}J(w_1,w_2,...,w_n)$ to denote the derivative of $J$ with respect to **one of the variables $w_i$**
  - **"partial derivative notation"**

Andrew Ng.: 

> In a way that I don't think this notational convention is actually necessary. **So for this class, I'm just going to use this notation $\partial $ everywhere**.

Next:

> let's take a look at how you can compute derivatives in a neural network. To do so, we need to take a look at something called a **computation graph**

## 11.2 Computation graph

The computation graph is a key idea in deep learning, and it is also how programming frameworks like TensorFlow, **automatic compute derivatives** of your neural networks.

Illustrate the concept of computation graph with a small neural network example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117105747.png"/>

- Linear regression model: takes input $x$, applies a linear activation function and outputs $a$ (**Basically just linear regression, but expressed as a neural network**) 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117104638.png"/>

  - Then the cost function: $J(w,b) = \frac{1}{2}(a-y)^2$

    > View the cost function as **a function of the parameters $w$ and $b$**

  - We have only one training example:

    $ x=2$ and ground truth label $y = 2$

  - Parameters of this network are:

    $w=2$ and $b=8$

- How cost function $J(w,b)$ is computed step by step using a computation graph:

  1. we first need to compute $w \times x$ , this is $c$
  2. Then, compute $a$, by $a = c+b$ (here, $b=8$, this value is already given)
  3. Then, compute $a-y$ (according to the cost function), this is $d$
  4. last step, compute $\frac 1 2 d^2$ .

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117105511.png"/>

  This is "computation graph"

This computation graph shows the **forward prop** step of how we compute the output a of the neural network.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117105634.png"/>

The question now is, **how do we find the derivative of $J(w,b)$?**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117105719.png"/>

> whereas **forward prop** was a left to right calculation **>>>**, computing the derivatives will be a right to left calculation **<<<** , which is why it's called **back prop**
>
> <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117110019.png"/>

Start with the final computation node $J=\frac 1 2 d^2$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117110019.png"/>

- How would value of $J$ change with the change of value of $d$?

  $\frac {\partial J}{\partial d} = 2$ >>> **This is the first step of back prop.** 

- Next: what is $\frac {\partial J}{\partial a}$ ?

  $\frac {\partial J}{\partial a} = 2$ still. >>> second step of back prop.

  > Until now, these steps of computations are actually relying on the **chain rule** for calculus
  >
  > **Chain rule**: $\frac {\partial J}{\partial a} = \frac {\partial d}{\partial a} \times \frac {\partial J}{\partial d} = 1 \times 2$ 

- Next:  

  - $\frac {\partial J}{\partial c} = \frac {\partial a}{\partial c} \times \frac {\partial J}{\partial a} = 1 \times 2 = 2 $
  - $\frac {\partial J}{\partial b} = \frac {\partial a}{\partial b} \times \frac {\partial J}{\partial a} = 1 \times 2 = 2$

- Now, one **final step**:

   $\frac {\partial J}{\partial w} = \frac {\partial c}{\partial w} \times \frac {\partial J}{\partial c} = -2 \times 2 = -4$

So, what we've just done is, **manually carry out back prop in this computation graph** (written in green color):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117113036.png"/>

Let's double check what we just did:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117113517.png"/>

- $w$ goes up by $\epsilon$, $J$ goes down $4 \epsilon$, we did it right!

Why do we use the back prop algorithm to compute derivatives? 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117114410.png"/>

**Efficiency**.

- back prop sequence is a right-to-left calculation, that makes you know every single step. After the calculation, you know how does the change of every element affects final $J$.

- Why it's efficient? >>> Every derivative term is only needed to be **compute once, then to be stored**. 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240117114515.png"/>

  So, it turns out that if a computation graph has $N$ nodes, meaning your $N$ of these boxes and $P$ parameters, then **back prop allows us to compute all the derivatives in roughly $N+P$ steps, rather than $N\times P$ steps**.

To recap: How the computation graph takes all the steps of the calculation needed to compute the output of a neural network a as well as the cost function $J$:

1. Takes a step-by-step computations and breaks them into the different nodes of computation graph. 
2. Then uses a left-to-right computation **forward prop** to compute the **cost function $J$**. 
3. Then a right-to-left (**back propagation**) calculation to computes all the **derivatives**.

Next:

> Let's take these ideas and apply them to a **larger neural network**.

## 11.3 Larger Neural Network Example

Let's take a look at how the computation graph works on a larger neural network example.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240119102056.png"/>

First: compute this step by step

- To make the math more tractable, we're still using **one single training example**:

  $x=1, y=5$

- we use **ReLU** activation: $g(z) = max(0,z)$

- Parameters of this NN:

  - $w^{[1]}=2, b^{[1]}=0$
  - $w^{[2]}=3, b^{[2]}=1$

- The computation flow of this neural network:

  - Input $\vec x$ **>>>** 
  - $a^{[1]} = g(w^{[1]} \vec x + b^{1}) = w^{[1]} \vec x + b^{1}=2$ **>>>** 
  - $a^{[2]} = g(w^{[2]} a^{[1]} + b^{2})=w^{[2]} a^{[1]} + b^{2} = 7$

  **So, the final predicted $\hat y$ is $7$**

- Cost function (squreed error cost function): 

  - Error: the gap between ground true label $y$ and predicted $\hat y$ 
  - $J(w,b)=\frac 1 2 (a^{[2]}-y)^2 = \frac 1 2 (7-5)^2 = 2$

Next: write this process down in a computation graph, the do a back prop:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240119104423.png"/>

- "so the backprop procedure gives you a very efficient way to compute all of these derivatives, which you can then feed into the gradient descent algorithm or the Adam optimization algorithm, to then train the parameters of your neural network."

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240119104802.png"/>

  - This will be very efficient when the neural network becomes really large.

- My question: why last forward prop step calculates $\frac 1 2 (a^{[2]}-y)^2$ directly? Seems skipped some steps before that..

  - No, no skip. **That's just how it works**, after getting the last prodicted value, directly calculate squreed error and cost $J$.

> Many years ago, before the rise of frameworks like tensorflow and pytorch, researchers used to have to manually use calculus to compute the derivatives of the neural networks that they wanted to train. And so in modern program frameworks, you can specify forward prop and **have it take care of back prop** for you.

Thanks to the computation graph and these techniques for automatically carrying out derivative calculations (is sometimes called autodiff, for automatic differentiation).

## Practice Lab 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240122123238.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240122123536.png"/>

I can use only 2 lines of code to do this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240122123558.png"/>

insted of this (hint by teacher):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240122123637.png"/>

- The parameters have dimensions that are sized for a neural network with $25$ units in layer 1, $15$ units in layer 2 and $10$ output units in layer 3, one for each digit.

    - Recall that the dimensions of these parameters is determined as follows:
        - If network has $s_{in}$ units in a layer and $s_{out}$ units in the next layer, then 
            - **$W$ will be of dimension $s_{in} \times s_{out}$**.
            - **$b$ will be a vector with $s_{out}$ elements**
    
    - Therefore, the shapes of `W`, and `b`,  are 
        - layer1: The shape of `W1` is (400, 25) and the shape of `b1` is (25,)
        - layer2: The shape of `W2` is (25, 15) and the shape of `b2` is: (15,)
        - layer3: The shape of `W3` is (15, 10) and the shape of `b3` is: (10,)
>**Note:** The bias vector `b` could be represented as a 1-D (n,) or 2-D (n,1) array. Tensorflow utilizes a 1-D representation and this lab will maintain that convention: 

The largest output of this model is $2$, means the prediction is $2$:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240122150304.png"/>

model:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240122150338.png"/>

This model (not included softmax layer), simply outputs the possibilities of the input `X` to being each of these 10 values.

- If the problem only requires a selection, that is sufficient. Use NumPy [argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) to select it. 
- **If the problem requires a probability, a softmax is required**

```python
prediction_p = tf.nn.softmax(prediction)
```

Straightly apply to the output `prediction` of last model, that's it.

Run freely

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240122151049.png"/>

# 12 Advice for applying machine learning

## 12.1 Deciding what to try next

You now have a lot of powerful tools of machine learning (linear regression, logistic regression, even deep learning, or neural networks, next will be decision trees), but **how do you use these tools effectively**?

I've seen teams spend six months to build a machine learning system, that I think a more skilled team could have taken or done in just a couple of weeks.

The efficiency of how quickly you can get a machine learning system to work well, will depend to a large part on how well you can **repeatedly make good decisions about what to do next** in the course of a machine learning project.

> In this week, I hope to share with you a number of tips on how to make decisions about what to do next in machine learning project

Let's start with an example: 

- say you've implemented regularized linear regression to predict housing prices. But if you train the model, and find that **it makes unacceptably large errors in it's predictions**, what do you try next?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240122152248.png"/>

There are a lot of choices, but how can you choose he right one? 

In this week, you'll learn about how to carry out a set of **diagnostic**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240122152438.png"/>

- Some of these diagnostics will tell you: is it worth weeks, or even months collecting more training data (which may save you months of times).

Next:

> First, let's take a look at how to **evaluate** the performance of your learning algorithm.

## 12.2 Evaluating a model

If you've trained a machine learning model, then how do you evaluate that model's performance? 

Let's take the example of learning to predict **housing prices** as a function of the **size** $x$:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124005220.png"/>

- Model: fourth order polynomial ($x, x^2, x^3, x^4$)
  - We don't like this model very much, because even though the model fits the training data well, it probably fail to generalize to new examples.
  - Why? The curve is **too wiggly** and too perfectly fitted into all the training examples.
- However, when adding more features $x_1, x_2, x_3, x_4$ into the model, it is hard to plot (4 dimension graph in 2D paper?)

We need some more systematic way to evaluate how well your model is doing:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124010821.png"/>

- "split the training set into 2 subsets"
  - First part: 70% of the data, **training set**
  - Second part: 30% of the data, **test set**.
- So here, 7 training examples, and 3 test examples.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124013525.png"/>

1. Fit parameters by minimizing cost function $J(\vec w,b)$, including the regularization term
2. Then, compute the **test error** $J_{test}(\vec w,b)$ , remember this does not include that regularization term.
3. Another quantity often useful too: the **training error** (again, this does not include that regularization term).

If you got these two datasets to make some comparison, then if an overfitting occurs, the error in the test set will be very high 👇

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124013556.png"/>

This was regression with squared error cost. 

Now, let's take a look at how you apply this procedure to a **classification problem** 👇

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124014048.png"/>

Still three steps:

1. Fit parameters. Remember to include the regularization term.
2. Compute test error (no regularization term)
3. Compute train error (no regularization term)

There shouldn't be a huge gap between test error and  train error.

In the classification problem, insted of using the logistic loss to compute the test error and the training error, another more commonly used way it **to measure what the fraction of the test set, and the fraction of the training set that the algorithm has misclassified**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124014435.png"/>

So, now:

- $J_{test}(\vec w,b)$ is the fraction of the test set that had been misclassified
- $J_{train}(\vec w,b)$ is the fraction of the test set that had been misclassified

Compare these two, we can have a sense of our model's performance.

Next:

> if you're trying to predict housing prices, should you fit a straight line to your data, or fit a second order polynomial, or third order, fourth order polynomial? 
>
> It turns out that with one further refinement to the idea you saw in this video, you'll be able to have an algorithm help you to **automatically make that type of decision well**.

## 12.3 Model selection and training / cross validation / test sets

one thing we've seen before was: once the model's parameters $\vec w$ and $b$ have been fit to the training set, the training error **may not be a good indicator** of how well it will generalize to new examples.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124105353.png"/>

And $J_{test}(\vec w,b)$ is a better indicator of how well your model has done its job.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124111614.png"/>

- use $w^{<1>},b^{<1>}$ to $w^{<10>},b^{<10>}$  respectively to denote different degrees of polynomial in different functions. 

- Then, train every model on your training set, and test them (By computing $J_{test}$ on the test set)...

- Then: one thing you could try is, look at all of these J tests, and see which one gives you the lowest value (for example, $J_{test}(w^{<5>},b^{<5>})$), then you might decide to use this model in practice.

  Why now we don't need to report test error again? 

  > Because now $J_{test}(w^{<5>},b^{<5>})$ is likely to be an optimistic estimate of the generalization error, so  it is likely to be lower than the actual generalization error.
  >
  > "if you want to choose the parameter d using the test set, then the test set J test is now an overly optimistic, that is **lower than actual estimate of the generalization error**."

  It turns out too, that if you want to choose the parameter $d$ using the test set, then the test set $J_{test}$ is now an **overly optimistic error**.

- Conclusion: The procedure on this particular slide is **flawed** and I **don't recommend** using this.

To modify the procedure: insted of split data into 2 subsets, now we can split into **3 subsets: training set, cross-validation set, and test set**.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124114306.png"/>

- Introduce the "cross validation set": 
  - this is an extra dataset that we're going to use to cross check the validity or really the accuracy of different models.
  - people may call it ""validation set" or "development set", "dev set".
- The proportion may be 60%, 20% and 20% for these three respectively.

Onto these three subsets of the data: training set, cross-validation set, and test set, you can then compute the training error, the cross-validation error, and the test error using these three formulas:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124115652.png"/>

- Remember, whereas usual, none of these terms include the regularization term.

This is how you can then go about carrying out model selection: 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124122106.png"/>

The procedure are as follows:

1. Fit the parameters $w_1$, $b_1$ , from model $d=1$ all the way down to the model, $d=10$
2. Instead of evaluating this on your test set, you will instead **evaluate these parameters on your cross-validation sets** and compute $J_{cv}$ , from order of <1> to <10>.
3. Then, in order to choose a model, you will look at which model has the **lowest cross-validation error**
   - For example, if $J_{cv}(w^{<4>},b^{<4>})$ reports the lowest cross validation error, then pick this model for your further application.
4. Finally, if you want to report out an estimate of the generalization error of how well this model will do on new data (not given and test yet), you will do so using **test set**, to compute $J_{test}$
   - Now, $J_{test}$ will be a **fair estimate of the generalization error** of this model (because up until now, it had not been used to do any training, **it's entirely new for our model**.)

Conclude:

> This gives a better procedure for model selection, and it lets you **automatically make a decision** like what order polynomial to choose for your linear regression model.

This also very useful to choose other types of models, for example, neural network architecture:

> If you are fitting a model for handwritten digit recognition, you might consider three models like these 👇 (small, larger, largest)

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240124123805.png"/>

1. Step 1: train all three of these models and end up with parameters $w^{<1>}, b^{<1>}$, $w^{<2>}, b^{<2>}$ and $w^{<3>}, b^{<3>}$

2. Step 2: evaluate the neural networks performance using $J_{cv}$ , compute cross-validation error👇

   > Since this is a classification problem, $J_{cv}$ the most common choice would be to compute this as the fraction of cross-validation examples

   if $J_{cv}(w^{<2>}, b^{<2>})$ has the lowest dev error, we pick the second neural network and use parameters trained on this model.

3. Finally, if you want to report out an estimate of the **generalization error**, you then use the **test set** to estimate how well the chosen model $(w^{<2>}, b^{<2>})$ will do.

It's considered **best practice** in machine learning that if you have to make decisions about your model, such as fitting parameters or choosing the model architecture.

Andrew Ng.:

> I use this all the time to automatically choose what model to use for a given machine learning application.

Next:

> let's dive more deeply into examples of some diagnostics. The most powerful diagnostic that I know of and that I used for a lot of machine learning applications is one called **bias and variance**.

## In Optional Lab:

Later in this lab, you will be adding polynomial terms so your input features will indeed have different ranges, so **feature scaling** is needed in this case. you will use the [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) class from scikit-learn. This computes the z-score of your inputs. As a refresher, the z-score is given by the equation:

$$
z = \frac{x - \mu}{\sigma}
$$
An *important* thing to note when using the z-score is you have to use the mean and standard deviation of the **training set** when scaling the cross validation set (make sure is using the same $z = \frac{x - \mu}{\sigma}$  to predict the value, this will make sure the cross validation set is scaled in same way >>> the accurate prediction).

you can try adding polynomial features to see if you can get a better performance. The code will mostly be the same but with a few extra preprocessing steps. Let's see that below.

**Note**:

- Remember to come back and figure out how last example's "Record the fraction of misclassified examples for the training set" step is working, thoroughly.

  ```python
  # Initialize lists that will contain the errors for each model
  nn_train_error = []
  nn_cv_error = []
  
  # Build the models
  models_bc = utils.build_models()
  
  # Loop over each model
  for model in models_bc:
      
      # Setup the loss and optimizer
      model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
      )
  
      print(f"Training {model.name}...")
  
      # Train the model
      model.fit(
          x_bc_train_scaled, y_bc_train,
          epochs=200,
          verbose=0
      )
      
      print("Done!\n")
      
      # Set the threshold for classification
      threshold = 0.5
      
      # Record the fraction of misclassified examples for the training set
      yhat = model.predict(x_bc_train_scaled)
      yhat = tf.math.sigmoid(yhat)
      yhat = np.where(yhat >= threshold, 1, 0)
      train_error = np.mean(yhat != y_bc_train)
      nn_train_error.append(train_error)
  
      # Record the fraction of misclassified examples for the cross validation set
      yhat = model.predict(x_bc_cv_scaled)
      yhat = tf.math.sigmoid(yhat)
      yhat = np.where(yhat >= threshold, 1, 0)
      cv_error = np.mean(yhat != y_bc_cv)
      nn_cv_error.append(cv_error)
  
  # Print the result
  for model_num in range(len(nn_train_error)):
      print(
          f"Model {model_num+1}: Training Set Classification Error: {nn_train_error[model_num]:.5f}, " +
          f"CV Set Classification Error: {nn_cv_error[model_num]:.5f}"
          )
  ```

- GPT answer:

  “# Record the fraction of misclassified examples for the training set”这一步是在计算训练集上的误分类率。我将逐行解释这部分代码的作用：

  1. `yhat = model.predict(x_bc_train_scaled)`：使用训练好的模型对训练数据集（已经缩放过的特征）`x_bc_train_scaled`进行预测。`yhat`包含模型的原始预测值，通常为logits（即未经过sigmoid变换的原始输出）。
  2. `yhat = tf.math.sigmoid(yhat)`：应用sigmoid函数转换`yhat`。因为模型的输出是logits，sigmoid函数将这些值映射到(0, 1)区间，表示概率。
  3. `yhat = np.where(yhat >= threshold, 1, 0)`：应用阈值（这里是0.5）来将概率转换为类别标签。如果预测的概率大于或等于0.5，则预测为类别1，否则为类别0。
  4. `train_error = np.mean(yhat != y_bc_train)`：计算误分类率。`yhat != y_bc_train`创建了一个布尔数组，其中True表示预测值与实际值不同（即误分类的情况）。`np.mean`计算这个数组中True值的比例，即误分类率。
  5. `nn_train_error.append(train_error)`：将计算出的误分类率添加到列表`nn_train_error`中，该列表用于存储所有模型在训练集上的误分类率。

# 13 Bias and variance

## 13.1 Disgnosing bias and variance

When I'm training a machine learning model, it pretty much never works that well the first time. Key to the process of building machine learning system is **how to decide what to do next in order to improve his performance**.

looking at the **bias and variance** of a learning algorithm gives you very good guidance on what to try next.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240202170544.png"/>

- **High bias**: underfit
- **High variance**: overfit
- "Just right"

Instead of trying to look at plots like this, a more systematic way to diagnose if your algorithm has high bias or high variance will be:

> to look at the performance of your algorithm on the **training set** and on the **cross validation set**

Like this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240202171658.png"/>

- **first example (d=1)**: $J_{train}$ is **high** because there are actually pretty large errors between the examples and the actual predictions of the model. $J_{cv}$ would also be **high**.

  > High **bias**. 
  >
  > (unsuccessful in fitting the training set)

- **second example (d=2)**: $J_{train}$ and $J_{cv}$ are very low

  > **Good!**

- **third example (d=4)**: $J_{train}$ is low, and $J_{cv}$ significantly higer than $J_{train}$

  > High **variance**. 
  >
  > (it does much better on data it has seen than on data it has not seen. )

⬆️ This gives you a sense, even if you can't plot $f(x)$ 

how $J_{cv}$ and $J_{train}$ varies as a function of the degree of the polynomial:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240202173047.png"/>

- **horizontal axis**: $d$, the degree of polynomial that we're fitting to the data.
- **vertical axis**: $J_{train}$ and $J_{cv}$ >>> corresponded to what we observed before.

To summarize: how do you diagnose **bias** and **variance** in your learning algorithm?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240202173938.png"/>

- the key indicator of **high bias** will be $J_{train}$ is high

- the key indicator for **high-variance** will be if $J_{cv}$ is **much greater** than $J_{train}$

- in some cases (espicially in neural network training), is possible to simultaneously have high bias and have high-variance ($J_{train}$ is high and the $J_{cv}$ is much larger).

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240202173852.png"/>

  some case like this: overfit in some part of the input but underfit in the rest.

The key takeaways are:

1. **high bias** means is not even doing well on the training set
2. **high variance** means, it does much worse on the cross validation set than the training set.

Next:

> let's take a look at how **regularization** effects the bias and variance of a learning algorithm. Because that will help you better understand when you should use regularization.

## 13.2 Regularization and bias/variance

let's take a look at how regularization, **specifically the choice of the regularization parameter $\lambda$ affects the bias and variance** and therefore the overall performance of the algorithm.

This will be helpful for when you want to choose a good value of $\lambda$ of the regularization parameter:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240202175235.png"/>

- originally set $\lambda = 10000$, then your $f(x)$ end up with a flat line (because now the algorithm is highly motivated to **keep these parameters $\vec w$ very small** -- close to 0) >>> $f(x)$ aprroximately equals to $b$ 

  > High bias

- Other extreme: small $\lambda$, $\lambda = 0$. In this case, **no regularization** >>> **overfits** the data.

  > High variance

- ✅ intermediate value for $\lambda$ 

So, how can we choose a good value of $\lambda$ ?

> **cross-validation** gives you a way to do so.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240203110832.png"/>

This procedure is similar to what you had seen for choosing the degree of polynomial:

- Try every possible value of $\lambda$ , each to fit the training set (minimize the cost function), then compute $J_{cv}$,
- If (for example) $J_{cv}(w^{<5>}, b^{<5>})$ is the lowest, then we pick $\lambda = 0.08$, and $(w^{<5>}, b^{<5>})$ as chosen parameters.
- Finally, after deciding the 5th $\lambda$, report the generalization error, $J_{test}(w^{<5>}, b^{<5>})$.

To further hone intuition about what this algorithm is doing, let's take a look at how **training error** and **cross validation error vary as a function of the parameter $\lambda$**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240203163800.png"/>

- Horizontal axis: value of $\lambda$ (Again)

- When $\lambda$ is small, then overfits the data (**high variance**), then $J_{train}$ becomes very **low**, however $J_{cv}$ very **high**.

- When $\lambda$ very large, **high bias** and underfits the data, then $J_{train}$ and $J_{cv}$ very high at the same time.

- With $\lambda$ goes up, the more weight is given to the regularization term, and thus the **less attention is paid to actually do well on the training set** (which means the $J_{train}$ **increases** as well). 

  so:

  > the more we trying to keep the parameters small, the less good a job it does on minimizing the training error.

- In cross validation set, between overfit and underfit, there'll be some **intermediate value** of $\lambda$ that causes the algorithm to **perform best**.

  > what cv set doing is to hopfully pick a $\lambda$ value that has low cross validation error, and this will hopefully correspond to a good model for your application.

- Compare this graph to "degree of polynomial as a function of $\lambda$" graph we used before:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240203164122.png"/>

  These looks a little bit like mirrored each other, because their "high bias" and "high variance" ends are really mirrored (not mathematically).

  > Higher order polynomial $d$ >>> more exquisite
  >
  > Higher parameters, higher $\lambda$ >>> more rough
  
  But they also helps to choose a good value of $d$ and good value of $\lambda$.

That's how the choice of regularization parameter $\lambda$ affects the bias and variance and overall performance of your algorithm.

Next:

> We kept saying "high $J_{train}$" or "much higher $J_{cv}$ than $J_{train}$. So, **what does these words "high" or "much higher" actually mean**?
>
> next video, where we'll look at how you can look at the numbers $J_{train}$ and $J_{cv}$ and judge if it's high or low, and it turns out that one further refinement of these ideas, that is, **establishing a baseline level of performance we're learning algorithm will make it much easier for you to look at these numbers**, $J_{train}$, $J_{cv}$, and judge if they are high or low.

## 13.3 Establishing a baseline level of performance

Look at some concrete numbers that $J_{train}$, $J_{cv}$ might be, by a Speech recognition example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240205115146.png"/>

- To train a speech recognition model, measuring the training error is necessary.

  > the **training error** means what's the percentage of audio clips in your training set that the algorithm **does not transcribe correctly** in its entirety.

  Here, training error $J_{train}$ is 10.8% >>> it transcribes it perfectly for 89.2% of your training set, 

  but makes some mistake in 10.8%.

- Then: the $J_{cv}$ here is 14.8%.

  > Seems pretty high!

- How can we know this error $J$ is high or low? 

  > when analyzing speech recognition, it's useful to also measure one other thing which is: **what is the human level of performance**? 

  human level performance achieves **just 10.6%** error..

  - So this algorithm did really well on training set!

- Now, the actual problem is with the cross validation set. The $J_{cv}$ is pretty much higher than $J_{train}$

  > So, this algorithm actually has more of a **variance problem** than a bias problem.

It turns out when judging if the training error is high, it is often useful to **establish a baseline level of performance**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240205115433.png"/>

Measurements:

1. Human level performance
2. Some competing algorithms performance (maybe a previous implementation that someone else has implemented or even a competitor's algorithm).
3. Guess based on prior experience

Then, the two different quantities to measure are:

1. the difference between training error and the baseline level
   - Large >>> **bias** problem
2. the gap between training error and cross-validation error
   - Large >>> **variance** problem

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240205120235.png"/>

⬆️ here:

- example 1 has a **high variance** problem (14.8% is much higher than 10.8%)
- example 2 has a **high bias** problem (15.0% is much higher than 10.6%)
- example 3 🤯 both **high bias** and **high variance**

Next:

> to further hone our intuition about how a learning algorithm is doing, there's one other thing that I found useful to think about which is the **learning curve**.

## 13.4 Learning curves

Learning curves are a way to help understand how your learning algorithm is doing **as a function of the amount of experience it has** (experience: e.g. the number of training example it has):

for example, apply a quadratic function (second ordered polynomial) to a training set:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240213182800.png"/>

- **horizontal axis**: training set size $m_{train}$

- **vertical axis**: error, either $J_{cv}$ or $J_{train}$ 

- The figure for training error is interesting (as the training set size gets bigger, the training set error actually **increases**) 

  **Why**?

  - Beacuse with everytime training set gets a little bit bigger, **it gets a little bit harder to fit all training examples perfectly**.

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240213182356.png"/>

    > when you have a very small number of training examples like one or two or even three, is relatively easy to get 0 or very small training error, but when you have a larger training set is harder for quadratic function to fit all the training examples perfectly.

- the cross-validation error will be typically **higher** than the training error.

  **Why**?

  * beacuse you fit parameters to the training set, not cross-validation set.

What the learning curves will look like for an algorithm with high bias versus one with high variance? 👇

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240213191241.png"/>

- with increasing of $m$, the curve of training error may start to **flatten out** (plateau) after a while.

  Beacuse:

  > average training error dosen't change that much about the straight line you're fitting.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240213185444.png"/>

  > It's just **too simple** a model to be fitting into this much data.

- similarly, the learning curve for **cross validation error**, beyond a certain point, flattened as well.

- And he baseline level (human-level performance) is lower than $J_{cv}$ and $J_{train}$ , means **underfitting** >>>  **High bias**

- What will heppen if the training set expands? >>> still be the same. Both curve **still be the flat line** with the experience of new training examples.

  > So, if a learning algorithm has high bias, getting more training data will not by itself hope that much.

High variance (fits the training set very well, but **doesn't generalize**): 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240213191935.png"/>

- A huge gap between the $J_{cv}$ curve and the $J_{train}$ curve. Because the algorithm is doing much better on the training set than on the cross validation set.

- The loss of baseline level of performance may higher than  $J_{train}$ but lower than $J_{cv}$ (another proof of "dosen't generalization")

- if increase the training set size, could **help a lot** in **high variance** scenario.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240213192002.png"/>

  if you were to add a lot more training examples and continue to fill the fourth-order polynomial, then you can just get a better fourth order polynomial fit to this data than this very wiggly curve up on top 👇

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240213192143.png"/>

Takeout:

> If you're building a machine learning application, you could plot the learning curves if you want, that is:
>
> - you can take different subsets of your training sets. 
> - f you have, say, 1,000 training examples:
>   - you could train a model on just 100 training examples and look at the training error and cross-validation error
>   - then train a model on 200 examples
>   - holding out 800 examples and just not using them for now, and plot J train and J cv and so on the repeats and plot out what the learning curve looks like.
>
> one **downside** is, it is computationally quite expensive to train so many different models using different size subsets of your training set.

Next:

> let's go back to our earlier example of if you've trained a model for housing price prediction, **how does bias and variance help you** decide what to do next?

## 13.5 Deciding what to try next revisited

This is the procedure I **routinely** do when I'm training a learning algorithm, often look at the training error and cross-validation error to try to decide if my algorithm has high bias or high variance. It turns out this will help you make better decisions about what to try next in order to improve the performance of your learning algorithm. 

Let's look at an example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240217120921.png"/>

What do you try next?

> each of these six items **either** helps fix a **high variance** or a **high bias** problem.

- **Get more training examples**: helpful in fixing high variance problem

  > the training error remains relatively flat even as the training set increases.
- **Try smaller sets of features**: high variance 

  > reduces noise >>> reduce overfitting
- **Try getting additional features**: high bias
- **Try adding polynomial features**: high bias
- **Try decreasing $\lambda$**: high bias
- **Try increasing $\lambda$**: high variance 

So:

- if you find that your algorithm has high variance, then the two main ways to fix that are:
  - get more training data
  - simplify your model
- High bias:
  - make your model more powerful (give them more flexibility to fit more complex or more wiggly functions)
  - Some ways to do that are to **give it additional features** or **add these polynomial features**, or to **decrease the regularization parameter $\lambda$**.

subsequently, after many years of work experience in a few different companies, he realized that bias and variance is one of those concepts that takes a short time to learn, but **takes a lifetime to master**.

Next:

> bias and variance also are very useful when thinking about how to train a neural network. In the next video, let's take a look at these concepts applied to **neural network training**.

## 13.6 Bias/variance and neural networks

One of the reasons that neural networks have been so successful is because neural networks, together with the idea of **big data** or hopefully having **large data sets**, it's given us new ways to **address both high bias and high variance**.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240218114415.png"/>

if you're fitting different order polynomials to a data set:

- **Trade off** between bias and variance:
  - if the model is too simple >>> high bias
  - if the model is too complex >>> high variance
- So, MLEs need to balance the complexity (e.g. degree of polynomial; value of $\lambda$) 

Neural networks offer us a way **out of this dilemma** of having to tradeoff bias and variance with some caveats:

> "Large neural networks almost always fits the training set well."

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240218121348.png"/>

- procedure:
  1. make your neural network bigger and bigger until it does well on the training set (by measuring the $J_{train}$)
  2. get more data until the $J_{cv}$ is acceptable.
  3. **Done**!
- That's why the rise of neural networks has been really assisted by the rise of very fast computers, including especially GPUs.

what if my neural network is **too big**? Will that create a high variance problem?

> "It turns out that a large neural network with well-chosen regularization, will usually do as well or **better than a smaller one**."

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240220121744.png"/>

- Even the risk of overfitting goes up significantly in larger neural networks, but if you were to **regularize this larger neural network appropriately**, then this larger neural network usually will do at least as well or better than the smaller one.
- it almost **never hurts** to go to a larger neural network so long as you regularized appropriately (main hurts: training will be more expensive).

How to apply regularization:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240218123923.png"/>

- Simply add `kernel_regularizer=L2(0.01)` to each layer, here, $\lambda$ is `0.01`. 
- You can choose different $\lambda$ for each layer, or for simplicity, same for each layer.

Two takeaways:

> 1. It hardly ever hurts to have a larger neural network so long as you regularize appropriately
> 2. so long as your training set **isn't** too large, then a neural network, especially large neural network is **often a low bias machine** 
