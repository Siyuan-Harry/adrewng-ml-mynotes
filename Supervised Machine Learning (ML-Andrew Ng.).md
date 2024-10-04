Preface: 

> That's machine learning. It's a science of getting computers to learn without being explicitly programmed.

> The only way we knew how to do these things was to have a machine learn to do it by by itself.

> This is sometimes called artificial general intelligence, or AGI.

- 由思远整理
- B站：何思远方Siyuan

> ©️Siyuan . Learning notes of Professor Andrew Ng.'s Machine learning specialization.

---

# 2 Machine Learning Overview

## 2.1 What is Machine Learning

The field of study that gives computers the ability to learn without being explicitly programmed.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501162037.png"/>

By far, the 2 most used types of learning algorithms today are **supervised learning** and **unsupervised learning.**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501162147.png"/>

This course contains :

- Mostly: Supervised Learning and Unsupervised Learning
- Biefly talk about: Reinforcement Learning
- The other thing we're going to spend a lot of time on in this specialization is **practical advice for applying learning algorithms.**

## 2.2 Supervised Learning Part.1

> I think 99% of the economic value created by machine learning today is through one type of machine learning, which is called **supervised learning.**

### 2.2.1 What is Spervised Learning?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501163132.png"/>

Explain:

- In short: Algorithms that learn x to y (input to output) mappings.
- The key characteristic of supervised learning is that **you give your learning algorithm examples to learn from that include the right answers,** by seeing correct pairs of input x and desired output label y that the learning algorithm eventually learns to take just the input alone without the output label and gives a reasonably accurate prediction or guess of the output.
- A procedure that computer learns to output the right answer(Y) by given the input(X).

### 2.2.2 The applications of Supervised Learning

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501173913.png"/>

### 2.2.3 Example: Housing price prediction(Regression)

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501175148.png"/>

Question: A friend wants to know **what's the price for their 750 square foot hulls.**

**Version 1: A straight line**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501174151.png"/>

👆It looks like your friend's house could be sold for maybe about 150,000 dollars?

**Version 2: A curve line**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501175349.png"/>

👆Well, your friend's house could be sold for close to 200,000 dollars.

How to choose whether the straight line or curve line? That's what we'll talk about later in this course.

### 2.2.4 Conclusion

What is supervised learning

- We give the algorithm a dataset (X and the right answer Y), and the task of this algorithm is to produce **more of these right answers.**

A type of Supervised Learning: **Regression**

- this housing price prediction is a particular type of supervised learning called ==Regression== (**predict a number** from infinitely many possible numbers, like this example).
- Another major type of supervised learning problem: **Classification.**

## 2.3 Supervised Learning Part.2

> There's a second major type of supervised learning algorithm called a **classification** algorithm. Let's take a look at what this means.

### 2.3.1 Example: Breast cancer detection (Classification)

Only 2 possible outputs (different from Regression):

- 0, benign
- 1, malignant

That's also what we're doing in MPB.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501182434.png"/>

Explain:

- Benign: good tumor. Malignant: bad tumor.
- The dataset contains size of tumor(Input) and the diagnosis by human doctor(Output).
- Algorithm learns through the dataset and finally, when input with the size of tumor, the Algorithms outputs the status of the tumor(benign or malignant).

Visualize like this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501182449.png"/>

We can also have 3 possible categories (represent as 0, 1, 2):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501191131.png"/>

Categories(classes) can be non-numeric. It can predict:

- Cat, dog
- a tumor is benign or malignant

But always, ==Classification== can only predict the input into **small limited number of categories,** as an output.

you can also use **more than one input value** to predict an output (just as my doubt):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501195942.png"/>

What the learning algorithm do is to find a boundary that seperates malignant and benign. 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501200055.png"/>

"My friends who worked on cancer detection use many additional inputs like the thickness of the tumor clump, uniformity of the cell size, uniformity of the cell shape, and so on."

### 2.3.2 Recap

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501200442.png"/>

## 2.4 Unsupervised Learning Part.1

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501200858.png"/>

> Don't let the name "unsupervised" fool you. Unsupervised learning is, I think, just as "super" as supervised learning.

The difference between Supervised and Unsupervised Learning:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501201027.png"/>

When doing unsupervised learning, we only give **the dataset without any labels(Y).**

The job of unsupervised learning algorithm is **to find some structure or some pattern** or just find something interesting in the ==unlabeled== data. 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501201356.png"/>

"Unsupervised":

- We don't supervise the algorithm to give some definite right answer for every input.

### 2.4.1 Clustering algorithm

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501201651.png"/>

Explanation:

- Places th unlabeled data into different clusters.

### 2.4.2 Example: Google news(Clustering)

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501202105.png"/>

"finding the articles that mention similar words and grouping them into clusters." **>>> Without human intervention.**

### 2.4.3 Example: DNA microarray(Clustering)

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230501202517.png"/>

We don't give the algorithms the dataset of right answer in advance, and algorithms classifies people(DNA microarray) into different groups with their DNA data **automatically.**

### 2.4.4 Example: Group customers(Clustering)

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230502201559.png"/>

The clustering algorithm helped Deeplearning team to know (in terms of learning motivation) what's the major groups of customers of Deeplearning. >>> "Finding the market segments"

### 2.4.5 Conclusion

main points:

- Clustering algorithm: a type of unsupervised learning algorithm that takes data without labels, and tries to automtically group them into clusters. >>> "Group similar data points together."

Other types of unsupervised learning: Go On...

## 2.5 Unsupervised Learning Part. 2

Reiterate of unsupervised learning and put forward another two Unsupervised learning algorithm: **Anomaly detection & Dimensinality reduction.**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230502202720.png"/>

Explanation:

- Anomaly detection: used to detect unusual events.
- Dimensinality reduction: Reduct the scale of dataset into smaller one with as little information loss.

Q&A little excercise:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230502203355.png"/>

## 2.6 Jupyter Notebooks

> For you to more deeply understand these concepts, I'd like to invite you in this class to see, run and maybe later write code yourself to implement these concepts.

### 2.6.1 Optional Labs

All you need to do is open it up and **just run the code we've provided.** 

By reading through and running the code in the optional labs, **you see how machine learning code runs** and you should be able to complete them relatively quickly just by **running it one line at a time from top to bottom.**

It's totally optional, **but better to try on** as it helps you to see what machine learning code actually look like and to get a deeper "feel" about machine learning. 

 ### 2.6.2 Pactice Labs

Starting next week there will also be some **practice labs** which would give you an opportunity to **write some of that codes yourself.**

### 2.6.3 Example: Optional Lab

2 different kinds of cells:

- Markdown cell
- Code cell

```markdown
A **coding sandbox** intergrated in the Coursera platform.
That's good...
Very friendly introduction, suitable for beginners!
```

> " look forward to seeing you in the next video where we'll take the supervised learning problem and start to flesh out our first supervised learning algorithm. **I hope that will be fun too and look forward to seeing you there**."

⚠️ Missing lecture

# 3 Linear Regression with One Viriable

> Linear regression: probably the most widely used learning algorithm in the world.

## 3.1 Part. 1

### 3.1.1 Example: House price predict

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230505172039.png"/>

Dataset: House sizes && House prices (Recently sell for)

Question: "My house is 1250 squarefeet. How much do u think I can get for this house?"

Solution: Build a linear regression model from this dataset. (blue line, 220k)

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230516163505.png"/>

Regression model Predicts **numbers**.

Classification model Predicts **categories**. (**Small number of possible outputs)**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230516163839.png"/>

### 3.1.2 Terminology

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230516164740.png"/>

Includes:

- Training set
- Input variable (Feature)
- Output variable (Target variable)
- Number of traning examples 
- single training examples
- i(th) training examples

## 3.2 Part. 2

Supervised learning algorithm will input the data set, and then **what exactly does it do and what does it output? Let's find out in this video.**

The framework of Regrassion model

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230516165814.png"/>

"using the model(function) f to predict a possible output y-hat from input x."

- `y-hat` is the prediction of `y`.
- `f` is the model.
- `x` is the input feature.
- `y` is the actual true value.
- `y-hat` is the output prediction, can be called as "estimated y".

How to represent `f` ?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230516172009.png"/>

- `f(X)` as the same meaning with `f(w,b)(X)`.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230516172201.png"/>

Here's what this function is doing:

- Making predictions for the value of y using a straight line function of x. 
- This is **linear regression with one variable**. (Another fancy name: **Univariate linear regression**)

"In a later video you also see a variation of regression where you want to make a prediction based not just on the size of a host, but on a bunch of other things that you may know about the house, such as a number of bedrooms and other features." >>> **Multivariate linear regression.**

When you're done with this video, there is another **Optional Lab:**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230517215652.png"/>

"You don't need to write any code, just review it, run the code and see what it does. That will show you how to define in python a straight line function."

"Cost function"

- In order for you to make this work, one of the most important things you have to do is **construct the cost function.** 
- The idea of a cost function is one of the most universal and important ideas in machine learning and is used in both linear regression and in training many of the most advanced AI models in the world.

## 3.3 Cost function formula

> The cost function will tell us how well the model is doing so that we can try to get it to do better.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230517221051.png"/>

- `w`,`b` are **parameters/coefficients/weights of the model**, can be also referred as "coefficients" or "weights", are the variables you can adjust during training in order to improve the model.

What does `w` and `b` do? like this 👇

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526161627.png"/>

Explaining:

- First example: b is called "y - intercept", and y is always a constant value (flat line).
- Second and third example: slope is 0.5.

Here explains relations between $\hat{y}$ and every specific $x^{(i)}$ , $y^{(i)}$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526162224.png"/>

Now, how to find `w`,`b` ?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526164433.png"/>

To do that, we're gonna construct a **cost function** (here is squared error cost function):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526170733.png"/>

Explaining:

- The cost function takes the prediction $\hat{y}$ and compares it to the target `y` by taking $\hat{y}$ minus `y` (which is called `error`), and then, square this `error` and get $\frac{1}{2}$ of average of every error for $\hat{y}^{(i)}$ , that's it!

$$
J(w,b)=\frac{1}{2m}\sum_{i=1}^m(\hat{y}^{(i)}-y^{(i)})^2
$$

But what is the cost function really computing? Go ahead.

## 3.4 Cost function intuition

Here is what we've got right now:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526172153.png"/>

And now our goal is to minimize $J(w,b)$ (`J` as the function of w and b). Using simplified function $J(w)$ to show this process: 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526175422.png"/>

If w=1, then the $f_w(x)$ and the $J(w)$ are like that:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526192852.png"/>

If w=0.5, then the $f_w(x)$ and the $J(w)$ are like that:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526194003.png"/>

If w=0, now we can get the whole picture of cost function  $J(w)$ .

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526194213.png"/>

then, how to choose the value of $w$ ?

- Choose $w$ to minimize $J(w)$ .
- $w=1$ , that's it!

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230526194514.png"/>

"Now in this video we work through our example with a simplified problem using only w. In the next video, let's visualize what the cost function looks like for the full version of linear regression using both w and b."

## 3.5 Visualizing the cost function

Here is what we've seen so far:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527155516.png"/>

Now we put visualization of cost function a bit further, which creates not from $f_w(x)$ but from $f_{w,b}(x)$ (more complex version of original model). With one more parameter $b$ join in the model, the cost function of which can not be visualized only in 2D graph, but should be in 3D space.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527160458.png"/>

👇The $J(w,b)$ function actually look like:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527160545.png"/>

Extraordinary little joke in class:

> To me, this looks like a super, maybe because I'm a little bit hungry. Or maybe to you it looks like a curved dinner plate or a hammock. Actually, that sounds relaxing too. And there's your coconut drink. Maybe when you're done with this course, you should treat yourself to a vacation and relax in a hammock like this.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527160654.png"/>

Understanding this 3D graph:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527160950.png"/>

How can we express the 3D graph like cost function $J(w,b)$ in 2D plane? >>> **Using contour plot.**

This is the contour plot of Mount Fuji:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527161528.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527161600.png"/>

Each of these ovals in the graph👇, also called **ellipses**, shows is the set of points on the 3D surface which are at the exact same height (in this case, they are same value of $J$ in cost function).

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527161745.png"/>

Understanding these three graphs:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527161914.png"/>

Intro to every graph:

- Upper right: the contour plot of cost function $J(w,b)$ . In the plot, three points (green, yellow, blue) are pointed out, they.
- Upper left: graph of model $f_{w,b}(x)$ , which inclues data set (red cross) and three models of prdiction (green, yellow, blue) corresponded to the points pointed out in the contour plot. In this graph, we can see directly all 3 of which are pretty bad for predicting housing prices in this case. 
- Bottom: 3D plot of cost function $J(w,b)$ . In the plot, three points (green, yellow, blue) are pointed out, which also corresponded to three models in upper left graph and three points in upper right graph. ==This graph is available to try on in the optional lab.==

Now, the bottom of the "bowl" where the minimum of $J$ is this point:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527163803.png"/>

> In the next video, let's visualize some specific choices of $(w,b)$ in a linear regression model. So you can see how these different choices affect the straight line you're fitting to the data. Let's go on to the next video.

## 3.6 Visualization examples

In this example, we can see the mapping relation between 2 graphs:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527165225.png"/>

To explain: 

- $J(w,b)$ , one point is pointed out: $w=-0.15$ and $b=800$ . 
- This value of $w$ and $b$ specifies one model: $f_{w,b}(x)=-0.15x+800$ , which is a straight line (this line is not a good fit).

Now we adjust the value of $(w,b)$ in the cost function:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527170249.png"/>

This looks a little bit better than previous one.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527170417.png"/>

Worse..

And when we get to the (almost) center point of  $J(w,b)$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527170624.png"/>

This is a good fit.

**Optional Lab:**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527171602.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527171728.png"/>

Learning outcomes of the optional lab:

- The cost is a measure of how accurate the model is on the training data.
- procedure: compute the total cost >>> 

- You can automate the process of optimizing 𝑤 and 𝑏 using gradient descent.



An short introduction of gradient descent algorithm:

- "Automatically finding the values of parameters, then give you the best fit line that minimizes the cost function $j$."
- And variations on gradient descent are used to train not just linear regression, but some of the **biggest and most complex models in all of AI.**
- One of the most important algorithms in machine learning.

# 4 Training Linear Regression

## 4.1 Gradient Descent

> Grading descent is used all over the place in machine learning, not just for linear regression, **but for training.** For example, some of the most advanced neuro network models, also called **deep learning models.** Deep learning models are something you learn about in the second course. So learning this 2 of grading descent will set you up with one of the most important building blocks in machine learning.

Here's an overview of what we'll do with gradient descent:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527174328.png"/>

"Gradient descent is an algorithm that you can use to try to **minimize any function**, not just a cost function for linear regression."

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527174456.png"/>

Some functions may have more than 1 minimum, and this is an example of "have more than 1 minimum": 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527174728.png"/>

Now we can see how "gradient descent" works:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527175532.png"/>

Describe what happens:

- you're physically standing at this hill, and your goal is to start up here and get to the bottom of one of these valleys as efficiently as possible.
- Firstly, you're going to spin around 360 degrees and look around and choose a direction of first tiny little step downhill. Mathematically, this is **the direction of steepest descent.**
- Then, you move forward very little, and repeat finding direction >>> move forward...**until you get to the deepest valley of the graph.**

An interesting property in gradient descent (blue poly line):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527180244.png"/>

that means:

- a little bit of change in the starting point could cause a huge difference in the final value (this time, you start from where just a couple of steps different from starting point of first time, could make you go into another different valley).
- The bottoms of both the first and the second valleys are called **local minima.**

## 4.2 Implementing Gradient Descent

This is what gradient descent algorithm look like mathematically, first look at the algorithm of parameter $w$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527180730.png"/>

Explaining:

- " $=$ " here is an **assignment operator**. Here means assign the new value of $w$ to old $w$ . In other words, this makes the expression updating the parameter $w$ continuously (notice that the assignment operator in programming is different from truth assertion in math).
- " $\alpha$ " is called the **learning rate**. Usually a small postive number between 0 to 1. **"controls how big of a step you take downhill"**. We'll come back later to delve more deeply into how to choose a good learning race alpha.
- " $$\frac{\partial}{\partial{w}}J(w,b)$$ " is a **derivative term** of the cost function $J$ , which tells you **which direction to go** in order to go downhill most efficiently and (in combination with the learning rate $\alpha$ , it also determines **the size of the steps** you want to take downhill.

Now, we can also see what the algorithm of parameter $b$ look like:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527183416.png"/>

Remind one subtle detail:

- For gradient descent you want to **simultaneously update w and b,** meaning you want to **update both parameters at the same time.** 

So let's take a look at what this means:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527184359.png"/>

The incorrect one is because the update procedure. The difference between them:

- Left one: This value of $w$ in `tmp_w`  is from before $w$ gets updated here. Notice that the pre-update w is what goes into the derivative term in `tmp_b`.
- Right one: Wrong. Because $w$ goes into the `tmp_b` expression after `tmp_w` assigned its new value to $w$ , which means that the $w$ is not updating with $b$ simultaneously.

- Andrew: Just **stick to the correct simultaneous update** (left one) and not use this incorrect version on the right.

Preview of next video:

> Coming up in the next video, we'll go over derivatives together and you come away with the intuition and knowledge you need to be able to implement and apply gradient descent yourself.

## 4.3 Gradient discent intuition

Here's the gradient descent algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527190555.png"/>

What we're going to focus on now is get more intuition about:

1. What this **learning rate** is doing
2. What this **derivative** is doing.
3. Why when multiplied together like this, **it results in updates to parameters $w$ and $b$ ,** that makes sense?

In order to do this, we use simplified model again:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527192808.png"/>

This slide shows: 

-  " $$\frac{\partial}{\partial{w}}J(w)$$ " Take the deriptive of the function, and minus by $w$ , means: $w$ minus the *slope* of current point you're stand on, which make you go to the right direction (depending on whether the slope is positive or negative). 
-  Genius design ! 

Preview of next video:

>  One other key quantity in the grading descent album is the **learning rate $\alpha$.** How do you choose $\alpha$ ? What happens if it's too small or what happens if it's too big? 
>
> In the next video, let's take a deeper look at the parameter $\alpha$ to help build intuitions about what it does.

## 4.4 Learning Rate

The object of 4.4: understanding what the learning rate $\alpha$ is doing.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230528205536.png"/>

Let's see what could happen if the learning rate $\alpha$ is either too small or if it is too large: 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230528211454.png"/>

The answers are: 

- if $\alpha$ is too small: gradient descent will work, but it will be too slow.
- if $\alpha$ is too large: gradient descent may overshoot and never reach the minimum (in other words, Fall to converge, or the say, "diverge").

Another question: What if your parameter $w$ is already at a local minimum? like this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230528211707.png"/>

Explainning:

- Then the further gradient denscent steps do nothing. $w$ will be the same (It's easy to understand through the expression).

This also explains why grading descent can reach a local minimum even with a **fixed** learning rate $\alpha$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230528212615.png"/>

Which means:

- As we get near a local minimum, gradient descent will **automatically take smaller steps** because of $$\frac{\partial}{\partial{w}}J(w)$$ multiplys with $\alpha$.
- Genius design of this algorithm! 

Right now (In my perspective) Maybe this is what machine learning in general is doing: 

- For a multiparametric specific job, create a cost function $J(w)$ of which and use gradient decrease algorithm to minimize $J(w)$ , in order to get the best parameter value.

In next video: 

> Putting together grading descent with this cost function >>> That will give you your first learning algorithm, the linear regression algorithm.

## 4.5 Gradient Descent for Linear Regression

Intro:

> In this video, we're going to put it all together and use the square error cost function for the linear regression model with gradient descent. This will allow us to train the linear regression model to fit a straight line to our training data.

Now we have these components constructing the gradient descent algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230528225616.png"/>

Notice that:
$$
\frac{1}{m}\sum^{m}_{i=1}(f_{w,b}(x^{(i)})-y^{(i)})x^{i}
$$
is calculated with calculus from cost function:
$$
J(w,b)=\frac{1}{2m}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})^2
$$


Here is the calculation process (calculate the derivative term using calculus):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529154329.png"/>

 We can now carry out these 2 expressions and plug them into the algorithm, like this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529154758.png"/>

 How gredient descent works (in linear regression):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529160328.png"/>

Explaining:

- The cost function is a **convex function**: bowl-shaped function that has a single minimum (local minimum also global minimum).

In next video, we'll see this algorithm in action.

## 4.6 Running Gradient Descent

Let's see what this algorithm actually is doing. Here it goes:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529161110.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529161208.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529161227.png"/>

So that's gradient descent and we're going to use this to fit a model to the housing data!

This gradient descent process is called batch gradient descent:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529161634.png"/>

Explaining:

- **Batch gradient descent** is looking at the entire batch of training examples at each update ==(sum up all the errors and get an new value to the parameter)==.
- It turns out that there are other versions of gradient descent that do not look at the entire training set, but instead **looks at smaller subsets of the training data at each update step.** 

Informations about optional lab and practice quizzes:

- In the optional lab that follows this video, you see a **review of the gradient descent algorithm** as well as **how to implement it in code.** 
- You also see a plot that **shows how the cost decreases** as you continue training more iterations. 
- And you also see a **contour plot**, seeing how the cost gets closer to the global minimum as gradient descent finds better and better values for the parameters $w$ and $b$.
- I hope you also check out the **practice quizzes**, which are a nice way that you can double check your own understanding of the concepts is also totally fine if you don't get them all right the first time. And you can also take the quizzes multiple times until you get the score that you want.

My qustion:

- **From where at the cost function that gradient descent starts?** 

GPT answer of this question:

-  the initial point of gradient descent can be set randomly or by using predefined values based on prior knowledge or heuristics. There is no fixed rule or tradition for setting the initial point, and the choice of initialization method can depend on the specific problem and the characteristics of the data.

  Here are **a few commonly used initialization methods in practice**:

  1. **Zero Initialization**: In some cases, setting all parameters to zero is a simple and common choice for initialization. However, this method may not always be suitable, especially if the cost function is not convex, as it may result in the algorithm getting stuck in local optima.
  2. **Random Initialization**: Another common approach is to randomly initialize the parameters within a small range. This helps to break the symmetry and avoid the algorithm being biased towards a specific direction. Random initialization is often used when training deep neural networks to prevent all neurons from learning the same features simultaneously.
  3. **Heuristic Initialization**: Sometimes, domain knowledge or heuristics can be used to set initial values. For example, if you know that the target variable is likely to have a certain range or if you have prior information about the problem, you can initialize the parameters accordingly. This approach can help guide the optimization process in a specific direction.
  4. **Pretrained Initialization**: In transfer learning or when using pre-trained models, the initial point can be set by loading the parameters from a previously trained model. This is often done when the pre-trained model has been trained on a similar task or dataset, and the initial values can provide a good starting point for further fine-tuning.

- It's worth mentioning that the choice of initialization can have an impact on the convergence speed and the final solution of the optimization process. Sometimes, experimentation with different initialization methods is necessary to find the best results.

Notes from optional lab:

- In this lab, you will: **automate the process of optimizing 𝑤 and 𝑏** using gradient descent.

Next week:

- We'll learn to make linear regression much more powerful.
- Instead of one feature like size of a house, you learn how to get it to work with lots of features. 
- You also learn how to get it to fit non linear curves.

# 5 Linear Regression with Multiple Variables

## 5.1 Multiple Features

Here is one feature condition:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529164401.png"/>

Here is multiple features condition:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529172530.png"/>

Notations:

-  $X_j$ is $j^{th}$ feature, like, $X_1$ is feature `size in feet^2` .
- $n$ is number of features.
- $\vec{x}^{(i)}$ are features of $i^{th}$ trainning sample, such as,  the $\vec{x}^{(4)}$ refers to $4^{th}$ sample, which can be write as $[852,2,1,36]$, includes: 
  - size in feet^2: 852
  - number of bedrooms: 2
  - Number of floors: 1
  - Age of home in years: 36
-  ${x}^{(i)}_{j}$ is value of feature $j$ in $i^{th}$ trainning sample. For example,  ${x}^{(2)}_{4}$ refers to the value of 4th feature in 2nd sample, which is 40, and means this home aged 40 years.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529175624.png"/>

Here are some more examples:

- $\vec{x}^{(2)}=[1415,3,2,40]$
- ${x}^{(2)}_{3}=2$

What the model will look like:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529180457.png"/>

How to understand (look at the example in the red box):

- 80k is the base price of the house.
- The price of the house will increase 0.1k once the size of the house increase 1 squarefeet.
- ...
- The price of the house will decrease 2k once the age of the house increase 1 year.

So here 👇 is the definition of the the model with $n$ features.
$$
f_{w,b}(x)=w_1x_1+w_2x_2+\dots+w_nx_n+b
$$
Next, rewrite the model in a simpler but equivelant way: seeing $w$ and $x$ each as a vector:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529183458.png"/>

New more compact form of model (which is called **multiple linear regression**): 
$$
f_{w,b}(x)=\vec{w}\cdot\vec{x}+b
$$
Then (in my perspective) what algorithm is doing is finding the angle and position of the vector $w$ , and the value of $b$, to make $wx+b$ become the best fit of the training sample.

- This is how linear algebra knowledge apply to use...

That's it for linear regression with multiple features, which is also called **multiple linear regression.**

## 5.2 Vectorization

When implementing a learning algorithm, using vectorization will both make your code shorter and also make it run much more efficiently. 

How vectorization is better:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230531202947.png"/>

This exapmle means:

Without vectorization, your code is like this:

```Python
f = 0
for j in range (0,n):
  f = f + w[j] * x[j]
f = f + b
```

When implemented vectorization (Using `numpy` in Python to create arrays representing vectors):

```Python
w = np.array([1.0,2.5,-3.3])
b = 4
x = np.array([10,20,30])
f = np.dot(w,x) + b
```

The reason that the vectorized implementation is much faster is behind the scenes is the numpy dot function is able to use parallel hardware (GPU) in your computer. 

To recap:

> Vectorization makes your code shorter, so hopefully easier to write and easier for you or others to read. And it also makes it run much faster.

## 5.3 Vectorization Part 2

Intro:

> I remember when I first Learned about vectorization, I spent many hours on my computer taking an unvectorized version of an algorithm, running it, see how long I ran, and then running a vectorized version of the code and seeing how much faster that ran. And I just spent hours playing with that. And it frankly drew my mind that the same algorithm vectorized, would run so much faster. It felt almost like a magic trick to me.
> In this video, let's figure out how this magic trick really works.

This shows the difference between vectorized and unvectorized algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230531204834.png"/>

It's difference between 2 steps and 16 steps.

A concrete example of how this helps in multiple linear regression:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230531205319.png"/>

In gradient descent algorithm, right implementation of vectorization will like:

```Python
w = np.array([0.5,1.3,...,3.4])
d = np.array([0.3,0.2,...,0.4])
w = w - 0.1 * d
```

and this can solve lot of time (espicially in large project).

This is optional lab:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230531212852.png"/>

## 5.4 Gradient Descent for Multiple Regression

Implementing gradient descent for multiple linear regression with vectorization:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230601184537.png"/>

What it will look like when implementing gradient descent:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230601185354.png"/>

Notation:

- $X$ is feature, and $w$ is the param of this feature, like: 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230529180457.png"/>

-  ${x}^{(i)}_{j}$ is value of feature $j$ in $i^{th}$ trainning sample. 
- $j$ equals one ($w_1x_1$) actually means the gradient descent algorithm for 1st feature (because $X_j$ is $j^{th}$ feature).
- Now $x$ and $w$ are vectors, which means that from $w_1x_1$ to $w_nx_n$ can be presented all in one expression. 

An alternative way for finding $w$ and $b$ for linear regression which was called the normal equation:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230601195234.png"/>

> "Don't worry about the details of how the normal equation works. Just be aware that some machine learning libraries may use this complicated method in the backend to solve for $w$ and $b$."

And don't forget the optional lab:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230601195257.png"/>

What happens during the process:

```python
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                           
        err = (np.dot(X[i], w) + b) - y[i] # this is a scalar, namely a number  
        for j in range(n):                       
            dj_dw[j] = dj_dw[j] + err * X[i, j]  # update each value of w for each feature in one run
				dj_db = dj_db + err                        
    dj_dw = dj_dw / m                            
    dj_db = dj_db / m                                
    return dj_db, dj_dw
```

During **one gredient**, what happens is:

1. for each training example, compute the *error* of the whole expression $f_{w,b}(x)=w_1x_1+w_2x_2+\dots+w_nx_n+b$

2. compute the `dj_dw[j]`, namely `dj_dw` for every feature's parameter `w` in this gredient. So the `dj_dw` here is a vector, contains the right direction and step length for every `w`.

   - Note the `dj_dw` is originally defined as `dj_dw = np.zeros((n,))`, so the code just plus the every `dj_dw` value to the right position, and through looping over each training example, accumulate each `dj_dw` and give us an ultimate `dj_dw` for this specific iteration.

   - Also, this step is carried out in *multidimensional vector space*, where every parameter `w`s and `b` work together to go to the right direction of the cost function $J(\vec w,b)$ - this cost function is plotted in a *multidimensional space*.
   - How the right direction is pointed out? the original information comes from the `err` (the gap between the predicted and true values). We integrate `err` into the `dj_dw` and `dj_db`, then they combined together, gives the right direction of gradient descent. 
   - each $x_j^i$ , intergrated into the `dj_dw` derivative term, helps to optimize the $w_j$.

3. after getting `dj_dw`, compute `dj_db`, and finally divided by `m` to output the real `dj_dw` & `dj_db` in this gradient. This is how every parameter is optimized in that specific point.

Now, move on to next video:

> Let's go on to the next video to see those little tricks that will help you make multiple linear regression work much better.

## 5.5 Feature Scaling Part 1

Intro:

> In this video you see a technique called feature scaling that will enable gradient descent to run much faster.

Let's fist look at size of a feature (the range of its value number):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230601214437.png"/>

⬆️ For this example, $x_1$ takes on a relatively **large** range of values and $x_2$ takes on a relatively **small** range of values.

Now, for a house which size is 2000 and number of bedrooms is 5, what do you think are reasonable sized parameters $w_1$, $w_2$ ?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230601214945.png"/>

We notice an interesting phenomenon:

- When the possible values of a feature $x$ are small, like the number of bedrooms, then a reasonable value for its parameter $w$ will be relatively large, like 50 in this case. 

So how does this relate to gradient descent? See the scatter plot of the features and contour plot in parameters (horizontal axis <<< size, and vertical axis <<< numberOfBedrooms):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230601222407.png"/>

Explaining:

- A very small change to $w_1$ can have a very large impact on the estimated price, and thus a very large impact on the cost $J$. Because $w_1$ tends to be multiplied by a very large number (the size in square feet).
- By contrast, it takes a much larger change in $w_2$ in order to change the predictions as much.

What does this leaves us? 👇

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230601224249.png"/>



This rescaled contour plot (bottom right) more like circles and less tall and skinny. So that gradient descent can find a much more direct path to the global minimum.

To recap:

> When you have different features that take on very **different ranges of values**.It can cause gradient descent to run slowly. 
>
> But rescaling the different features so they all take on **comparable range of values** can speed up great and descent significantly. 

## 5.6 Feature Scaling Part 2

How to scale features?

Method 1, divided by maximum of range:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230615172630.png"/>

Method 2, mean normalization:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230615174332.png"/>

- $\mu_1$ is mean of the $x_1$ on thrining set.
- to calculate the mean normalization of $x_1$, use this formula: $\frac{x_1-\mu_1}{max-min}$

Method 3, Z-score normalization:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230615175210.png"/>

- formula: $\frac{x_1-\mu_1}{\sigma}$ , just like what I learned from Probability Theory course..

When to apply feature scaling:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230615180816.png"/>

> "There's almost never any harm to carrying out feature rescaling. So when in doubt, I encourage you to just carry it out."

## 5.7 Checking Gradient Descent for Conergence

When running gradient descent, how can you tell if it is converging?

Here is something you should do to make sure that gradientdescent is working well:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622173326.png"/>

- Learning curve: plot the cost function $J$ and plot the value of $J$ at each iteration at gradient descent.
- Plotting this can make us see when gardient descent converged. 
- Notice that this learning curve shown above means that the gradient descent algorithm works well.

Another way to decide when your model is done training is with an **automatic convergence test**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622173511.png"/>

- sometimes finding appropriate $\epsilon$ is too difficult, so choose to look at the learnig curve.

## 5.8 Choosing the learning rate

Problem of too large learning rate:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622174049.png"/>

To fix this, you can use smaller learning rate $\alpha$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622174239.png"/>

Or if leanrning curve keeps going up, that may suggests there is bug in the code (mistakenly written the "$-$" to "$+$"):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622174456.png"/>

How to debug: 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622174719.png"/>

- Set $\alpha$ to be a very small number, to see if the learning curve is going down (only works as a debugging step, because it is not a efficient way to train the algorithm).

You can adjust the learning rate $\alpha$ ike this, to find the appropriate $\alpha$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622174909.png"/>

Try this optional lab:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622175022.png"/>

Next:

> There are a couple more ideas that can use to make multiple linear regression much more powerful. And that is choosing custom features which will also allow you to fit curves, not just a straight line to your data. 

## 5.9 Feature engineering

What is feature engineering:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622175944.png"/>

Creating new features, **in order to make it easier for the learning algorithm to make accurate predictions.** ('sometimes by defining new features, you might be able to get a much better model.')

Why intruducing feature engineering here:

> it turns out that there's one flavor of feature engineering that allow you to fit not just straight lines, but curves, non linear functions to your data. Let's move on.

## 5.10 Polynomial regression

In order to fit curves to the dataset,  we can use quedratic function, but which will go down with size of the house rises:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622180752.png"/>

So we apply cubic function (not just x squared but x cubed):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622181045.png"/>

All of these above are examples of polynomial regression (because you took your optional feature x and raised it to the power of 2 or 3, or any other power).

if you create features so that these powers like the square of the original features like this then **feature scaling** becomes increasingly important:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622181254.png"/>

How to choose features, like $x^2$ or $x^{\frac{1}{2}}$ ?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230622205825.png"/>

This content is in following courses.

# 6 Classification

## 6.1 Motivations

Some examples of classification problems:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230625173838.png"/>

 This type of classification problem where there are only 2 possible outputs, called as **binary classification**.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230625174358.png"/>

- The output includes 2 classed/categories (the terms "class" and "category" are relatively interchangeably here).
- Use the bool value (number 0 and 1) to represent the anser.

So how do you build **a classification algorithm**? See this example of tumor detection algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230625175226.png"/>

- (Thinking in linear regression way) you could put a **threshold** in value between 0 to 1, like 0.5, and let comparison with threshold to decide whether the output is positive or negative.

But now let's make some change:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230625180045.png"/>

- linear regression model could make misclassified output due to specially destributed training set (by moving the **decision boundary** in a wrong way). 
- You learn more about the **decision boundary** in the next video. 
- You also learn about an algorithm called **logistic regression**: output always between (0,1) and will avoid this problem. And although the name contains "regression", the algorithm actually used for classification.

More about optional lab:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230625180112.png"/>

- In the upcoming optional lab, you will get to take a look at what happens when you try to use linear regression for classification. Sometimes you get lucky and it may work, but often it will not work well.
- You also see an interactive plot that attempts to classify between 2 categories.

## 6.2 Logistic Regression

> This is something that I use all the time in my work.

logistic regression will end up doing is fit a curve that looks like this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230701112611.png"/>

**sigmoid function**, which is also referred to as the logistic function. Look like this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230701112744.png"/>

The formula looks like this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230701112816.png"/>

- Basic principle about this function: when $z$ is large, $g(z)$, that is a sigmoid function of $z$, is going to be very close to $1$. 

Now let's build up to the logistic regression algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230701114013.png"/>

- Like, applying the value of the linear function as the argument of the logistic function. **Overlaying the logistic function to the linear function.**

How to interpret the output of logistic regression:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230701114329.png"/>

- This function outputs the probability that class is $1$.

In the Optional Lab, see how does it look like in code:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230701114422.png"/>

Go on to learn more about Logistic Regression!

## 6.3 Decision Boundary

to recap:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230702132257.png"/>

Then, the goal of the function $f_{\vec{w},b}(\vec{x})$ is to predict when the output should be $1$ or $0$ .

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230702134026.png"/>

- Setting a threshold of value $0.5$ , then judging the output of  $f_{\vec{w},b}(\vec{x})$ is or not bigger than $0.5$ .

- Returning the function: **$f_{\vec{w},b}(\vec{x})$ is equal to $g(z)$** . So $g(z)$ greater than or equal to 0.5 whenver $z$ is greater than or equal to $0$ >>> **$\vec{w} \cdot \vec{x} + b$ greater than or equal to $0$.**

- Conversely, when $\vec{w} \cdot \vec{x} + b$  is less than 0, the algorithm predicts y is 0.

Now let's visualize how the model makes predictions:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230702193319.png"/>

- **Decision boundary**: a line that is almost nutual about whether $y = 1$ or $y = 0$ , which is, $\vec{w} \cdot \vec{x} + b = 0$ . For this example, the specific decision boundary is $x_1 + x_2-3 = 0$ .
- If the features $x$ are to the right of this line, logistic regression would predict $1$, and to the left of this line, logistic regression would predict $0$.
- This is an example of logistic regression when the parameters $w_1 = 1$ , $w_2 = -1$ , and $b = -3$ .

Let's look at a more complex example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230702193823.png"/>

We can apply the polynomial regression into this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230702194129.png"/>

- When parameters $w_1 = 1$ , $w_2 = -1$ , and $b = -1$ , the decision boundary turns out to be this circle. 
- The formula of which is $x_1^{2} + x_2^{2} = 1$ .

We can come up with even more complex decision boundaries (you can do so by having even higher order polynomial terms):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230702194405.png"/>

See more in upcoming optional lab:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230702194429.png"/>

Next:

> Let's take a look at how you can actually train a logistic regression model. We'll start by looking at the cost function for this regression and after that, figure out how to apply gradient descent to it. Let's go on to the next video.

# 7 Cost Function

## 7.1 Cost Function for Logistic Regression

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230706140027.png"/>

- Use $m$ to denote the quantity of training examples, each training example has multiple features. Use $n$ to denote features. 
- The logistic model is $f_{\vec{w},b}(\vec{x})$ .
- The question is : **given this training set, how can you choose $\vec w$ and $b$ now?**
- Basic knowledge about cost function: when it comes to the linear regression, each point of the cost function corresponds to a model with specific parameters $w$ and $b$ ( $w$ and $b$ are the fixed coefficients that presents a line). And when gradient descent get to the global minimum of the cost function, we've got the best parameters $w$ and $b$ which presents a best line to fit the training set.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230706141224.png"/>

- The old form of cost function is only suitable for the convex function (bowl shape, contains only one local minimum, which is also global minimum). **But** now in logistic regression, we're facing the **non-convex function.** 
- The *squred error cost function* doesn't work well for logistic regression.
- Solution: construct a **logistic loss function**, to make the cost function convex again.

**Logistic loss function:**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230706141417.png"/>

Intuition about what this loss function is doing:

- The loss function measures how well you're doing on **one training example.** Simply add a layer of "-log" upon $f_{\vec{w},b}(\vec{x}^{(i)})$ 
- Cost function: by summing up the losses on all of the training examples (By contrast, cost function measures how well you're doing on the **entire training set**).

See the plot, to understand why this function makes sense:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230706142507.png"/>

When the **TRUE VALUE** is 1, the loss function look like this:

- The output of $f_{\vec{w},b}(\vec{x}^{(i)})$ is always **between 0 and 1**, so the loss function really works on this Domain of Definition.
- See the zoomed plot (DoD) on the left: when $y^{(i)}=1$, the loss will be almost ultimate large when prediction $\hat y$ is close to 0. By contrast, when prediction $\hat y$  is 1(accurate prediction), the loss should be 0 .
- So, the loss function compares **every output value** of logistic regression **with the sample**, and output the loss (difference) between them. Value 0 of loss denotes that the prediction is perfectly accurate, and as the value of loss get higher, the accuracy drops.

The second part of the loss function ( $y^{(i)} = 0$ ), which means when **TRUE VALUE** is 0：

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230919151004.png"/>

- If predicted $\hat y$ (the same as $f_{\vec{w},b}(\vec{x}^{(i)})$ ) is 0, then accurate prediction, the loss is 0.
- Otherwise, the loss would be infinitely large if the predicted $\hat y$ approaching 1.

Conclusion: 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230919151642.png"/>

In the optional lab: see what logistic cost function look like

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230919151743.png"/>

Learning from optional Lab:

- Differences between linear regression and logistic regression: 

  - Linear regression

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527155516.png"/>

  - Logistic regression

    **Model**   $$f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$$ 

    **Loss function**
    $$
    \begin{equation}
      loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \begin{cases}
    
    \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=1$}\\
    
    \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=0$}
    \end{cases}
    \end{equation}
    $$

- Simplified loss function: 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230919154846.png"/>

- "a cost function can be produced that incorporates the loss from all the examples. This will be the topic of the next lab."

Keep going!

## 7.2 Simplified Cost Function

> In this video, you'll see a slightly simpler way to write out the loss and cost functions so that the implementation can be **a bit simpler** when we get to grading descent for fitting the parameters of a logistic regression model. 

First to see simplified loss function:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230919155259.png"/>

- Relatively easy, eliminates half of this function when $y^{(i)}$ is 0 or 1.

Using this simplified loss function, let's write out the cost function:

- The cost j is the average loss across the entire training set $m$ .

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230919155725.png"/>

 Here 👆 is the cost function.

But, **why do we choose this particular function** when there could be tons of other cost functions we could have chosen? 

Explaining:

- This particular cost function is derived from statistics using a statistical principle called **maximum likelihood estimation**, which is an idea from statistics on how to efficiently find parameters for different models.
- And this cost function has the nice property that it is convex.
- Don't worry about learning the details of maximum likelihood. It's just a deeper rationale and justification behind this particular cost function.

### Optional Lab

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230919160240.png"/>

- The upcoming optional lab will show you how the logistic cross function is implemented in code.
- You will implement this later in the practice lab at the end of the week.

Notes

- Differences between linear regression and logistic regression: 

  - Linear regression

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230527155516.png"/>

  - Logistic regression

    **Model**   $$f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$$ 

    **Loss function** 
    $$
    \begin{equation}
      loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \begin{cases}
    
    \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=1$}\\
    
    \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=0$}
    \end{cases}
    \end{equation}
    $$
    **Cost function** $$ J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}$$

    

    Related point: 
    $$
    \begin{align}
      f_{\mathbf{w},b}(\mathbf{x^{(i)}}) &= g(z^{(i)})\tag{3} \\
      z^{(i)} &= \mathbf{w} \cdot \mathbf{x}^{(i)}+ b\tag{4} \\
      g(z^{(i)}) &= \frac{1}{1+e^{-z^{(i)}}}\tag{5} 
    \end{align}
    $$

  - However, the construct methods of cost function in linear regression and logistic regression are **completely different routes**.

With the simplified cross function, we're now ready to jump into **applying gradient descent to to logistic regression**. Let's go see that in the next video.

## 7.3 Gradient Descent Implementation

> To fit the parameters of a logistic regression model, we're going to try to find the values of the parameters w and b that minimize the cost function $J_{(w,b)}$

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230921172957.png"/>

- Given the tumor size $x$, the function should estimate the probability that the label $y$ is 1.

How to find a good choice of the parameters $w$ and $b$ : still gradient descent

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230921175210.png"/>

- $j$ goes from 1 to $n$, while $n$ is the umber of features.
- $\frac{\partial}{\partial{w_j}}J(\vec{w},b)$ can be written as $\frac{1}{m}\sum_{i=1}^m(f_{\vec{w},b}(\vec x ^{(i)})-y^{i})x_j^{(i)}$ as the output of the derivative calculation. Calculate by hand.
  - $x_j^{(i)}$ is the $j_{th}$ feature of the training example $i$.
- $\frac{\partial}{\partial{b}}J(\vec{w},b)$ is  $\frac{1}{m}\sum_{i=1}^m(f_{\vec{w},b}(\vec x ^{(i)})-y^{i})$ , the difference from upper is this output does not contain $x_j^{(i)}$.

The gradient descent algrorithm of logistic regression is here: (they look exactly like the linear regression!)

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230921175946.png"/>

- Difference: the function $f_{\vec w,b}(\vec x)$ has changed.
  - $f_{\mathbf{w},b}(x^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target.
  - For a logistic regression model 
        $z = \mathbf{w} \cdot \mathbf{x} + b$ 
        $f_{\mathbf{w},b}(x) = g(z)$ 
        where $g(z)$ is the sigmoid function: 
        $g(z) = \frac{1}{1+e^{-z}}$  
- Same:
  - You can just apply the same method for logistic regression to make sure it also converges
  - You can also use **vectorization** to make gradient descent run faster for logistic regression (can see the code in **optional labs**).
  - Use feature scaling, scaling all the features to take on similar range of values,  to help gradient descent to converge faster.

See more details in optional labs!

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230921180053.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230921180136.png"/>

- there'll be a nice set of animated plots that show gradient descent in action. 
- There will be another optional lab after that, which is short and sweet, but also very useful because I'll show you how to use the popular **sklearn library** in Python to train the logistic regression model for classification.

### Learning in optional lab

The relationship between cost function and gradient descent:

* the cost function is simply the accumulation of all the losses for every feature of every training example.
* the gradient gives a direction for every dimension of every feature ($x_j$) to move on (for batch gradient descent, by accumulating all the training examples in one run, to get a final direction for this step, for every feature). 

`def gradient_descent` : 

$$\begin{align*}
&\text{repeat until convergence:} \; \lbrace \\
&  \; \; \;w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1} \\ 
&  \; \; \;  \; \;b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
&\rbrace
\end{align*}$$

`def compute_gradient_logistic(X, y, w, b)` :

$$\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{2} \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3} 
\end{align*}$$

Note: 为什么Logistic regression的代价函数求导完后结果是 (2) 和 (3) 这两个式子？

GPT回复：

```markdown
为了理解为什么代价函数 \( J(\mathbf{w},b) \) 对参数 \( w_j \) 和 \( b \) 的偏导数是给定的形式，我们需要从代价函数的定义出发，逐步进行求导。

首先，代价函数 \( J(\mathbf{w},b) \) 定义为：
\[ J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \]

在logistic回归中，假设函数 \( f_{\mathbf{w},b}(\mathbf{x}) \) 通常定义为：
\[ f_{\mathbf{w},b}(\mathbf{x}) = \sigma(\mathbf{w} \cdot \mathbf{x} + b) \]
其中，\( \sigma(z) \) 是sigmoid函数，定义为：
\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

对于二分类问题，常用的损失函数是交叉熵损失函数：
\[ loss(h, y) = -y \log(h) - (1-y) \log(1-h) \]
其中 \( h = f_{\mathbf{w},b}(\mathbf{x}) \)。

代价函数可以写成：
\[ J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ -y^{(i)} \log(f_{\mathbf{w},b}(\mathbf{x}^{(i)})) - (1-y^{(i)}) \log(1-f_{\mathbf{w},b}(\mathbf{x}^{(i)})) \right] \]

为了求偏导数，我们需要对 \( J(\mathbf{w},b) \) 分别对 \( w_j \) 和 \( b \) 求导。

### 对 \( w_j \) 求导
首先，我们计算 \( \frac{\partial J(\mathbf{w},b)}{\partial w_j} \)：
\[ \frac{\partial J(\mathbf{w},b)}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} \frac{\partial}{\partial w_j} \left[ -y^{(i)} \log(f_{\mathbf{w},b}(\mathbf{x}^{(i)})) - (1-y^{(i)}) \log(1-f_{\mathbf{w},b}(\mathbf{x}^{(i)})) \right] \]

我们需要使用链式法则来求导。首先，定义：
\[ a^{(i)} = \mathbf{w} \cdot \mathbf{x}^{(i)} + b \]
\[ h^{(i)} = f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \sigma(a^{(i)}) \]

对 \( w_j \) 求导：
\[ \frac{\partial h^{(i)}}{\partial w_j} = \frac{\partial \sigma(a^{(i)})}{\partial a^{(i)}} \cdot \frac{\partial a^{(i)}}{\partial w_j} \]
\[ \frac{\partial \sigma(a^{(i)})}{\partial a^{(i)}} = \sigma(a^{(i)}) (1 - \sigma(a^{(i)})) = h^{(i)} (1 - h^{(i)}) \]
\[ \frac{\partial a^{(i)}}{\partial w_j} = x_j^{(i)} \]

因此：
\[ \frac{\partial h^{(i)}}{\partial w_j} = h^{(i)} (1 - h^{(i)}) x_j^{(i)} \]

接下来，我们对损失函数的每一项求导：
\[ \frac{\partial}{\partial w_j} \left[ -y^{(i)} \log(h^{(i)}) - (1-y^{(i)}) \log(1-h^{(i)}) \right] \]
\[ = -y^{(i)} \frac{1}{h^{(i)}} \frac{\partial h^{(i)}}{\partial w_j} - (1-y^{(i)}) \frac{-1}{1-h^{(i)}} \frac{\partial h^{(i)}}{\partial w_j} \]
\[ = -y^{(i)} \frac{1}{h^{(i)}} h^{(i)} (1 - h^{(i)}) x_j^{(i)} + (1-y^{(i)}) \frac{1}{1-h^{(i)}} h^{(i)} (1 - h^{(i)}) x_j^{(i)} \]
\[ = \left( -y^{(i)} (1 - h^{(i)}) + (1-y^{(i)}) h^{(i)} \right) x_j^{(i)} \]
\[ = (h^{(i)} - y^{(i)}) x_j^{(i)} \]

将其代入总的偏导数表达式中：
\[ \frac{\partial J(\mathbf{w},b)}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} (h^{(i)} - y^{(i)}) x_j^{(i)} \]
\[ = \frac{1}{m} \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} \]

### 对 \( b \) 求导
类似地，我们计算 \( \frac{\partial J(\mathbf{w},b)}{\partial b} \)：
\[ \frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} \frac{\partial}{\partial b} \left[ -y^{(i)} \log(f_{\mathbf{w},b}(\mathbf{x}^{(i)})) - (1-y^{(i)}) \log(1-f_{\mathbf{w},b}(\mathbf{x}^{(i)})) \right] \]

同样使用链式法则：
\[ \frac{\partial h^{(i)}}{\partial b} = \frac{\partial \sigma(a^{(i)})}{\partial a^{(i)}} \cdot \frac{\partial a^{(i)}}{\partial b} \]
\[ \frac{\partial \sigma(a^{(i)})}{\partial a^{(i)}} = \sigma(a^{(i)}) (1 - \sigma(a^{(i)})) = h^{(i)} (1 - h^{(i)}) \]
\[ \frac{\partial a^{(i)}}{\partial b} = 1 \]

因此：
\[ \frac{\partial h^{(i)}}{\partial b} = h^{(i)} (1 - h^{(i)}) \]

对损失函数的每一项求导：
\[ \frac{\partial}{\partial b} \left[ -y^{(i)} \log(h^{(i)}) - (1-y^{(i)}) \log(1-h^{(i)}) \right] \]
\[ = -y^{(i)} \frac{1}{h^{(i)}} \frac{\partial h^{(i)}}{\partial b} - (1-y^{(i)}) \frac{-1}{1-h^{(i)}} \frac{\partial h^{(i)}}{\partial b} \]
\[ = -y^{(i)} \frac{1}{h^{(i)}} h^{(i)} (1 - h^{(i)}) + (1-y^{(i)}) \frac{1}{1-h^{(i)}} h^{(i)} (1 - h^{(i)}) \]
\[ = \left( -y^{(i)} (1 - h^{(i)}) + (1-y^{(i)}) h^{(i)} \right) \]
\[ = h^{(i)} - y^{(i)} \]

将其代入总的偏导数表达式中：
\[ \frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} (h^{(i)} - y^{(i)}) \]
\[ = \frac{1}{m} \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \]

因此，我们得到了两个偏导数的结果：
\[ \frac{\partial J(\mathbf{w},b)}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} \]
\[ \frac{\partial J(\mathbf{w},b)}
```



```python
#A function that inputs dataset X, target values y, and parameters w and b (where ).
#Outputs the (2) dj_dw and (3) dj_db

def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw
```

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/IMG_6638.jpg"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/IMG_6639.jpg"/>

​                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

---

# 8 Regularition to Reduce Overfitting

## 8.1 The Problem of Overfitting

> What I'd like to do in this video is to show you what is **over fitting**, as well as a closely related, almost opposite problem called **under fitting**. 
>
> And in the next videos after this, I'll share with you some techniques for accessing overfitting. In particular, there's a method called **regularization**,

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230927175937.png"/>

- High bias: a preconception making the algorithm fits the dataset poorly. 
  - e.g. assuming a linear function could fit the housing price trainning set.
- Generalize well: means to make good predictions even on brand new examples.
  - In the middle: just right! Neither under fit nor overfit.
- Overfitting: fits **too well**, the cost function can reach exactly 0 point, however this model **may not generalize to new examples**.
  - high variance: a just slightly different datasets, they could end up with totally different predictions or **highly variable** predictions.

So far 👆 we've looked at underfitting and overfitting for linear regression model.

Similarly, overfitting applies to classification as well:

> Here's a classification example with 2 features, $x_1$ and $x_2$, where $x_1$ is the **tumour size** in $x_2$ is the **age of patient**.



<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230927180638.png"/>



- the 3rd example: overly complex decision boundary, overfit, high decision boundary.

## 8.2 Addressing Overfitting

> Later in the specialization, we'll talk about debugging and diagnosing things that can go wrong with learning algorithms. You also learn about specific tools to recognize when overfitting and underfitting may be occurring.

Tool #1: Collect more training data.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230929203546.png"/>

- Left: overfitted house price prediction model.

- Right: getting more training examples to address the overfitting problem. (when the data is available, this can work really well.)

Tool #2: To see if you can use **fewer** features 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230929204438.png"/>

To explain:

- More feature refers to more "$x$" in the model, and these lot of polynomial features could make the function more complicated and the curve more distorted
- How to select: to use your intuition to choose what you think is the best set of features (here, what's **most relevant** for predicting the price).
  - one disadvantage: throwing away some of the information that you have about the houses.
-  Later in course 2, you also see some algorithms for automatically choosing the most appropriate set, the features they use for prediction task.

Tool #3 "**Regularization**", which we'll look at in even greater depth in the next video.

> "Just for myself, I use regularization all the time. "

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230929205303.png"/>

- What regularization do: set the parametrs of large features to a relatively small number, in order to eliminate affections of these.

  > "encourage the learning algorithm to shrink the values of the parameters without necessarily demanding that the parameter is set to exactly 0.
  >
  > And it turns out that even if you fit a higher order polynomial like this, so long as you can get the algorithm to use smaller parameter values, w one, w 2, w 3, w 4, **you end up with a curve that ends up fitting the training data much better.**"

- It lets you keep all of your features, but it just prevents the features from having a overly large effect

- "It kind of doesn't make a huge difference whether you regularize the **parameter $b$** as well. You could do so if you want or not"

To recap:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230929205417.png"/>

About optional lab:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230929205529.png"/>

- In the lab, you will see different examples of overfitting and adjust those examples by clicking on options in the plots. 
- You also be able to add your own data points by clicking on the plot and see how that changes the curve that is fit.
-  You can also try examples for both regression and classification. And you really change the degree of the polynomial to be x, x squared, x cubed and so on. 
- The lab also lets you play with 2 different options for accessing overfitting. You can add additional training data to reduce overfitting and you can also select which features to include or to exclude as another way to try to reduce over fitting.

##  8.3 Cost Function with Regularization

> "regularization" >>> “正则化”
>
> “带有正则化的代价函数”

We'll build on that intuition and develop a modified cost function for your learning algorithm they can use to actually apply regularization.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230930105723.png"/>

Explaining: 

- Let's say you were to modify the cost function and add to it 1000 times $w_3^2$ plus 1000 times $w_4^2$ .
- So when you minimize this function, you're going to end up with $w_3$ close to 0 and $w_4$ close to 0.

But more generally, the way that regularization tends to be implemented is if you have a lot of features, say **100 features**, you may not know which are the most important features and which ones to penalize...

So the way regularization is typically implemented is **to penalize all the $w_j$ parameters**. And it's possible to show that this will usually result in fitting **a smoother, simpler, less wiggly function that's less prone to overfitting.**

Example: having 100 features

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230930114822.png"/>

- let's penalize all of them a bit and shrink all of them by adding this new term (starts with $\frac {\lambda}{2m}$ ). 
  - the Greek alphabet **lambda** is also called a **regularization parameter**.
  - Similar to picking a learning rate $\alpha$, you now also have to choose a number for **lambda**.
- More details: 
  - both the first and second terms here are scaled by $\frac{1}{2m}$ . It turns out that by scaling both terms the same way, it becomes a little bit easier to choose a good value for **lambda**.
  - In particular, even if your training set size grows, the "scale" movement would make the same value of **lambda** that you have picked previously more likely to continue to work in a larger training set.
  - $m$ is the size of training set, the number of traning examples. 
  - $\frac {\lambda}{2m} b^2$ makes **very little difference** in practice.
- This would make a new cost function.

This new cost function:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230930114941.png"/>

- This new cost function would try to achieve 2 goals at the same time: **(1)** minimize the mean squared error to best fit the traningset, and **(2)** minimize the regularization term to keep $w_j$ small.

  > brilliant disign! Utilizes the feature of cost function... 

- And the $\lambda$ that you choose specifies the relative importance / the relative tradeoff / how you **balance between these 2 goals**.

Take a look at what different values of **lambda** will cause your learning algorithm to do (using linear regression model):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230930115907.png"/>

- One extreme: If lambda was 0, you end up fitting this overly wiggly, overly complex curve and it **overfits**.
- Another extreme: if lambda is $10^{10}$, the learning algorithm fits a horizontal straight line and **underfits**.
- When the value of lambda is **just right**, then hopefully you end up able to fit a forth all the polynomial, keeping all of these features, but with a function that look like the purple curve in the plot.

> In the next 2 videos, we'll flesh out how to apply regularization to linear regression and logistic regression and how to train these models with gradient descent. With that, you'll be able to avoid overfitting with both of these algorithms.

## 8.4 Regularized Linear Regression

> We'll figure out how to get gradient descent to work with regularized linear regression.

Cost function for regularized linear regression:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012021550.png"/>

And you'd like to find parameters $w$ and $b$ that minimize the regularized cost function:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012021737.png"/>

Applying gradient descent, previously:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012021940.png"/>

Now, the derivative terms of cost functions are different:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012023209.png"/>

- the derivative term of $b$ still remain the same.

So, this is what the code need to do:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012023410.png"/>

In order for you to get this algorithm to work, until now, this is all you need to know.

The rest of this video: 

- go over some optional material to convey a slightly deeper intuition about what this formula is actually doing.
- As well as chat briefly about how these directors are derived.

Getting deeper:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012024146.png"/>

- About gradient descent, the only change when you add regularization is that instead of $w_j$ being set to be equal to $w_j$ minus $\alpha$ times this "usual update" term, is now **$w$ times $(1-\alpha\frac{\lambda}{m})$ minus the "usual update".**
- So, what does  $(1-\alpha\frac{\lambda}{m})$ mean? 👇

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012024456.png"/>

- So what recognization is doing on every single iteration is you’re multiplied $w$ by a number slightly less than one. And that has the effect of shrinking the value of wj just a little bit.
- So this gives us another view on why regularization has the effect of **shrinking the parameters $w_j$ a little bit on every iteration**. And so that's how regularization works.

How these derivative turns were computed:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012025134.png"/>

- So this is why this expression is used to compute the gradient in regularized linear regression.

## 8.5 Regularized Logistic Regression

Here's the idea:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012025523.png"/>

- We saw earlier that logistic regression can be prone to overfitting if you fit it with **very high order polynomial features** like this.
- More generally, when you train logistic regression with a lot of features, whether polynomial features or some other features, there can be a higher risk of overfitting.

This was the regularized cost function for logistic regression (new regularization term added):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012025710.png"/>

How can you actually implement this? How can you actually minimize this cost function $J(w,b)$ that includes the regularization term?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012025833.png"/>

Add $\frac{\lambda}{m}w_j$ to the code (still, we don't have to regularize $b$):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012030033.png"/>

Final optional lab:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231012030216.png"/>

> "I know you've only been studying this stuff for a few weeks, but if you understand and can apply linear regression and logistic regression, that's actually all you need to create some very valuable applications."
>
> "Having said that, there's still many more exciting things to learn. In the second course of this specialization, you learn about neural networks, also called deep learning algorithms. The way your neural network gets built actually uses a lot of what you've already Learned, like cost functions and grading descent and sigmoid functions."
>
> "So again, congratulations on reaching the end of this 3rd and final week. Of course, one, I hope you have funded the labs and **I will see you in next week's material on neural networks.**"

Next:

- Understanding the specific codes in last optional lab.
- Finish the practice lab.
