

# 1 Welcome to the course

## 1.1 Welcome!

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240531220658.png"/>

1. **First week**: *Clustering* and *anomaly detection* are the two techniques that widely used by many companies today in commercial applications.
2. **Second week**: you learn *recommender systems*. This is the core technique of many content platforms or shopping websites.
3. **Third week**: you learn about *reinforcement learning*. the number of commercial applications of reinforcement learning is not nearly as large as the other two techniques, it  is exciting and is opening up a new frontier to what you can get learning algorithms to do.

# 2 Clustering

## 2.1 What is clustering?

In supervised learning:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240531221029.png"/>

- We have both the input features **X** and the labels for each example $y$ .

In unsupervised learning, there is no label $y$. So the plotted dataset looks like this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240603181223.png"/>

- We ask the algorithm to find some interesting structure about this data
- For example, "clustering", identifies the **groups** of the dataset (here, 2 groups).

Applications of clustering:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240603181447.png"/>

- **DNA Analysis**:  look at the genetic expression data from different individuals and try to group them into people that exhibit similar traits.
- **Astronomical data analysis**: astronomers using clustering to group bodies together to figure out which ones form one galaxy or which one form coherent structures in space.

## 2.2 K-means intuition

Start with an example with a dataset of *30 unlabeled training examples*:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240603183609.png"/>

K-means algorithm will very firstly guess the *cluster centroids* (the *centers* of the 2 clusters here you ask it to identify), here the blue and red X. Then it **repeatedly** do 2 things:

1. **assign points to cluster centroids**: it will go through all of these examples, $x^{(1)}$ through $x^{(30)}$, and check each of them for its distance from the two cluster centroids. Finally assign it to whichever of the cluster centroids it is closer to.

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240603183521.png"/>

   - **Note**: this divides all the data points into **two groups** for this example. 

2. **move cluster controids**: compute the position of center for each group of points, and then move the centroids to that position.

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240603183542.png"/>

And return to Step 1, and then Step 2 again, so on and so forth.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240603183651.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240603183717.png"/>

After a specific point, you find:

- If you keep on doing these 2 steps, then no more movement of centroids is happening.
- This means the algorithm did its job successfully (converged). 

## 2.3 K-means algorithm

Now let's write out the K-means algorithm in detail so that you'd be able to implement it for yourself.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604005516.png"/>

1. **Step 1**: Randomly initialize $K$ cluster centroids. Here, the position of $\mu_1$ is red cross and $\mu_2$ is blue cross.

2. **Step 2** (repeated): assign the data points to each cluster centroid. For example, assign the first training example $x^{(1)}$ to the *centroid 1*, then it means we set the $c^{(1)} = 1$ (this value **1** denotes the **first cluster centroid**, and this variable $c^{(1)}$ means the assigned centroid of the 1st training example).

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604004410.png"/>

   - We assign the points based on distance. Methamatically, the distance between two points is often written like this: $||x^{(i)}-\mu_k||$

     > This is also called **L2 norm** .

3. **Step 3** (repeated): to move the cluster centriods. For each of the cluster centroids, compute the average of each dimension of all the points assigned to this centroid, this gives us **the final position where the cluster centroid is going to be updated**.

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604004723.png"/>

   So this is the computing of new position of $\mu_1$: $\mu_1 = \frac 1 4 [x^{(1)} + x^{(5)} +x^{(6)}+x^{(10)}]$ . To be clearer, each of these x values are vectors with:

   -  **2** numbers in them (in this case), or
   - **n** numbers in them. if you have n features instead of 2. 

   so, $\mu$ will also have 2 numbers in it or **n** numbers in it correspondingly.

One corner case of this algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604005249.png"/>

- if a cluster has **0 training examples in it**, then the step 2 is not able to compute.
- The **most common way** to solve this is just eliminate that cluster and end up with $k-1$ clusters.
- If you really need that cluster, then an alternative would be to just **randomly reinitialize** that cluster centroid and hope that it gets assigned at least some points next time round.

It turns out that K-means is also frequently applied to data sets where the **clusters are not that well separated**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604005449.png"/>

- it will works just fine (right plot).
- the cluster centroids will give you a sense of what is the most representative height and weight that you will want your three t-shirt sizes to fit.

> But what this algorithm really doing, and do we think this algorithm will converge or they just keep on running forever and never converge? 

Next:

> To gain deeper intuition about the K-means algorithm and also see why we might hope this algorithm does converge, let's go on to the next video where you see that K-means is actually trying to optimize a specific cost function.

## 2.4 Optimization objective

the K-means algorithm that you saw in the last video is also **optimizing a specific cost function**. Though it is not gredient descent.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604111259.png"/>

- **Explain the new $\mu_{c^{(i)}}$** : this refers to a specific **cluster centroid.** For example, the $x^{(10)}$ example is assigned to the cluster centroid 2, then $\mu_{c^{(10)}}$ is the **cluster centroid 2**.

- **Explain the cost function**: the aim of this function is to minimize the overall distance of every **data points** to each **cluster centroid**. 

  > what the K means algorithm is doing is trying to find assignments of points of clusters centroid as well as find locations of clusters centroid that **minimizes the squared distance**.

- This function is also called "*distortion*" cost function.

Visually:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604111443.png"/>

what they will do on every step is try to:

1.  update the cluster assignments $c^{(1)}$ through $c^{(30)}$ in this example. 
2. Or update the positions of the cluster centralism, $\mu_1$ and $\mu_2$.

Now, take a deep look at this algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604142635.png"/>

> what the K-means algorithm is doing is trying to find assignments of points of clusters centroid as well as find locations of clusters centroid that *minimizes the squared distance*.

Here the two steps of the repeated part, are both corresponded with the cost function $J$.

1. **Update $c^{(i)}$**: Assign points to cluster centroids, by computing the distance and assign the point to the nearest $\mu$.

2. **Update $\mu_k$**: move $\mu$ to a new position to minimize this expression: $\frac 1 m \sum^{m}_{i=1}||x^{(i)}-\mu_{c^{(i)}}||^2$ . And why does a new $\mu$ in the central of all points **minimizes** the overall squred distance? see explaination below:

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604155257.png"/>

   - moving the centroid, to a medium point of two given training examples.
   - This makes the squared distance from **41** decrease to **25**.

To conclude:

1. The fact that the K-means algorithm is optimizing a cost function **J** means that, it is guaranteed to converge, that is on every single iteration, the distortion cost function should **go down** or **stay the same**.
2. It should never go up.
3. Also, if the cost function ever stops going down (or goes down very slowly), then it mostly means the computation has converged.
4. So, **computing the cost function is helpful helps you figure out if the algorithm has converged**.

Next:

> It turns out that there's one other very useful way to take advantage of the cost function, which is to use multiple different random initialization of the cluster centroid. If you do this, you can often find much better clusters using K-means.

## 2.5 Initializing K-means

In this video, we explore how to implement **Step 0** in the slide below 👇:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604160329.png"/>

**The most common way**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604161646.png"/>

1. randomly choose a set of training examples.

2. set all the cluster centroids initially on top of these randomly picked training examples.

But, depending on how you choose the random initial central centroids, K-means will end up picking a difference set of causes for your data set:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604162835.png"/>

- the clustering result on top is good, but others not good (stuck in local optimum). This is because they are initialized differently.

  > "with this less fortunate choice of random initialization, it had got stuck in a **local minimum**."

- To solve the problem: run the algorithm multiple times, compute the cost function $J$ for all three of these initializations, and pick one accoding to which gives you the lowest value of cost function $J$.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604162901.png"/>

To be more specific, if you want to use 100 random intializations for K-means:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604163349.png"/>

1. randomly intialize K-means, run the algorithm and compute the cost function (distortion). 

2. repeat this for 50-1000 times, based on your time and computation resources.

   > When I'm using this method, doing this somewhere between say 50 to 1000 times would be pretty common.

3. Finally, pick one cluster result that gives the lowest cost $J$.

it just causes K means to do a much better job minimizing the distortion cost function and finding a much better choice for the cluster centroids.

Next, we discuss:

> The question of how do you choose the number of clusters centroids? How do you choose the value of K? 

## 2.6 Choosing the number of clusters

It's really hard to say if this dataset contains 2, or 3 or 4 clusters.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604201157.png"/>

One method from adademic literature that may be referred to by others (Andrew dosen't use it himself) is **Elbow method**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604211047.png"/>

- In the examples shown above, we decide $k=3$, because after 3, the value of cost function decreases significantly slower.

- **However**, in reality, the right "K" is often truly ambiguous. And the cost function - K plot is often like this:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604211231.png"/>

  - Note that there is no obvoius elbow.

- And, don't just choose **k** to minimize the cost function **J**, because based on the increase of **k**, the cost function **J** will almost always go down.

So how do you choose value of k in practice?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240604214622.png"/>

- Evaluate K-means based on later (downstream) purpose.
- for example:
  - What I would do in this case is to run K-means with **K = 3** and **K = 5** and then look at these two solutions to see based on the trade-off between **better fit** versus **the extra cost of making more t-shirts**, to try to decide what makes sense for the t-shirt business. - to decide the value of K.
  - This trade-off also appears at the image compressing process. You must trade-off between **how good the image looks** versus **how much you can compress the image** to save the space, to manually decide the best value of **K**.

Next, lab and:

> move on to our second unsupervised learning algorithm, which is **anomaly detection**. "How do you look at the data set and find unusual or anomalous things in it."
>
> This turns out to be another, one of the most commercially important applications of unsupervised learning.

# 3 Anomaly detection

## 3.1 Finding unusual events

Anomaly detection algorithms look at an unlabeled dataset of normal events and thereby learns to detect or to raise a red flag for if there is an unusual or an anomalous event.

**Example**: using anomaly detection to detect possible problems with **aircraft engines** that were being manufactured.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240607224000.png"/>

1. collect several features of an aircraft engine, like "*heat generated by the working engine*", and "*vibration intensity of the engine*", etc. (to simplify, we only use these two features for now.)

   - So, if several examples with these features plotted, it should look like this: 

     <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240607222652.png"/>

     a bunch of examples, each of them has $x_1$ feature and $x_2$ feature so that they can be presented in this 2-D graph.

2. If a new example inputted in, then (for this 2-D example), we can visually tell if it is normal or not:

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240607223434.png"/>

So, how can an algorithm address this problem? - Using a technique called *density estimation*.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240608103302.png"/>

- Given a training set, the first thing you do is build a model of probability of $x$

  In other words:

  > The learning algorithm will try to figure out what are the values of the features $x_1$ and $x_2$ that have **high probability** and what are the values that are not.

- Concretely, the region in the middle will have high probability, and the outside region will have lower probability.

  > The details of the algorithm is what we'll see next.

- When the algorithm is given a new test example $x_{(test)}$, it will firstly compute the probability of this example $p(x_{test})$, and then decide by applying a threshold $\epsilon$.

- this $\epsilon$ is going to be a very small number, meaning that if $p(x_{test})$ is smaller than this number, then $x_{(test)}$ could be an anomaly.

Anomaly detection is widely used today.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240608105022.png"/>

1. **Fraud detection**. If you're running a website, you can model your user's behavior through these data (to a probability model):

   - How often does a user log in?
   - How many pages visited?
   - How many transactions were made?
   - How many posts (frequency)?
   - What is the typing speed of that user?

2. You can identify unusual users by checking which have $p(x)< \epsilon$ (must perform a*dditional checks)*

   > algorithms like this are routinely used today to try to find unusual or maybe slightly suspicious activity.
   >
   > Moreover, it's used both to find *fake accounts* and this type of algorithm is also used frequently to try to **identify financial fraud** such as if there's a very unusual pattern of purchases.

3. many factories were routinely use anomaly detection to **see if whatever they just manufactured** (airplane engine, printed circuit board, smartphone, motor...) **have any anomaly problem**.

4. **Monitor computers in a data center**. if ever a specific computer behaves very differently than other computers, it might be worth taking a look at that computer to see if something is wrong with it.

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240608104507.png"/>

Anomaly detection is one of those algorithms that is very widely used even though you don't seem to hear people talk about it that much.

Andrew:

> I remember the first time I worked on the commercial application of anomaly detection was when I was helping a telco company put in place anomaly detection to see when any one of the cell towers was behaving in an unusual way. 
>
> Because that probably meant there was something wrong with the cell tower and so they want to get a technician to take a look, so hopefully that helped more people get good cell phone coverage. 

Next:

> we'll talk about how you can build and get these algorithms to work for yourself. 
>
> In order to get anonymous detection algorithms to work, we'll need to use a **Gaussian distribution** to model the data p of x. 

## 3.2 Gaussian (Normal) Distribution

In order to apply anomaly detection, we're going to need to use the **Gaussian distribution**, which is also called the normal distribution or bell-shaped distribution.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240608185043.png"/>

- $x$ is a random number, and probability of $x$ is determined by a Gaussian with mean $\mu$ and variance $\sigma^2$. This the $p(x)$ looks like a *bell-shaped curve*.

- The $\mu$ determines the center of the curve. The width (standard deviation) of this curve is givern by the variance parameter $\sigma$.

  - $\sigma$ - standard deviation
  - $\sigma ^2$ - variance

- **What does this bell-shaped curve mean**: if you have 100 number drawn from this probability distribution, you might get a **histogram** that looks like this (vaguely bell-shaped):

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240608111859.png"/>

- The function of this curve is:
  $$
  p(x) = \frac 1 {\sqrt {2\pi} \sigma} e^{\frac {-(x-\mu)^2} {2 \sigma^2}}
  $$

Now let's look at a few examples of how changing $\mu$ and $\sigma$ will affect the *Gaussian distribution*:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240608185347.png"/>

When you're applying this to anomaly detection, here's what you have to do:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240608185534.png"/>

- Above is the dataset (and the plot of which) given. 

- The first step to carry out is **to fit a Gaussian distribution to this dataset**.

- We should compute $\mu$ and $\sigma$ mathematically:

  - $$
    \mu = \frac 1 m \sum^m_{i=1} x^{(i)}
    $$

  - $$
    \sigma^2 = \frac 1 m \sum^m_{i=1} (x^{(i)} - \mu)^2
    $$

  - In some cases we use $\frac 1 {m-1}$ instead of $\frac 1 m$, but practically it makes very little difference.

- These formulas for $\mu$ and $\sigma$ is technically called **maximum likelihood estimates** for $\mu$ and $\sigma$.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240608190716.png"/>

  * after you get the $\mu$ and $\sigma$, you get a pretty good fit of *Gaussian distribution* on the dataset.

- After you get the bell-shaped curve of this dataset, the next thing is pretty clear: compute $p(x)$ for the new test example $x$ and judge if the value is higher of lower than threshold.

- Now, for this given example, $x$ is a **number** (1 feature). But in reality, you usually have a lot of features, which makes $x$ become a *vector*.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240608190652.png"/>

## 3.3 Algorithm

Now, we're ready to build our anomaly detection algorithm. Let's dive in.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609080524.png"/>

- You have traning set $\vec x^{(1)}$ through $\vec x^{(m)}$, each example $x$ has 2 features $x_1$ and $x_2$ (so here, let $n=2$). But for many practical applications, **$n$ can be much larger**.

- This algorithm often works fine even that the $n$ features are not actually statistically independent.

- For the given dataset, we need to **model each feature** by computing the $\mu$ and $\sigma ^2$ for each of them.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609080325.png"/>

- Here is an example of how this algorithm computes $p(\vec x)$ when an engine becomes *too hot* and *too high in vibaration* at the same time: 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609080433.png"/>

This is how we build the algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609082022.png"/>

- **Firstly**, choose $n$ features $x_j$ that may be relevent. **Then** collect the data for these features on each example and compute the parameters of the model.

  - ***the vectoriazed way***: if you're to compute $\mu$. To see the $\mu$ as a **vector** (each element in the vector is the $\mu_j$ for each feature), and simply take the average of all features of all the training examples $\vec x$.

  - ***The normal way***: compute each $\mu_j$ and $\sigma ^2_j$ at a time.

  - This means for each of features $x_j$, you fit a *Gaussian distribution*:

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609081858.png"/>

- **Finally**, for a newly inputted test examples $x$, you compute $p(x)$ using continued multiplication of possibilities for every feature of $x$ - to see if $p(x)$ is anomaly.

  - here, $x_j$ is the features of the new example. The $\mu_j$ and $\sigma^2_j$ is what you computed from previous examples.

  - Even one features $p(x_j)$ is very small, then the result $p(x)$ becomes very small.

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609082007.png"/>

To visualize an example which contains only two features $x_1$ and $x_2$:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609083134.png"/>

- The plot for $p(x)$ is a 3D plot, which combines three graphs above together.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609082554.png"/>

  - The height of the "mountain" is the product of $p(x_1)$ and $p(x_2)$:  $p(x_1) \cdot p(x_2)$ 

- Pick $\epsilon = 0.02$ - now the $p(x_{test}^{(1)})$ is normal but $p(x_{test}^{(2)})$ is nearly impossible (anomaly).

Next:

> How do you choose the parameter $\epsilon$? And how do you know if your anomaly detection system is working well?
>
> let's dive a little bit more deeply into the process of developing and evaluating the performance of an anomaly detection system.

## 3.4 Developing and evaluating an anomaly detection system

if you can have a way to evaluate a system, even as it's being developed, you'll be able to make decisions and change the system and improve it much more quickly.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609103812.png"/>

- it's important that if you have a small dataset of anomalous examples, and create a **cross-validation set** including them.
  - a few examples $y=1$
  - a lot of examples $y=0$ (normal)

Illustrate this with the aircraft engine example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609104548.png"/>

- pay attention to the proportion of different types of examples that distrubuted in different dataset.
  - training: 6000
  - cv: 2000 - 10
  - test: 2000 - 10
- we can use cv set to **tune** the parameter $\epsilon$ and feature $x_j$ 
- **An alternative way**: *no test set*. (if you only have 2 flawed engines, this really makes sense to only create cv set) - though it's making it more possible to overfit.

Take a closer look on how to evaluate your algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609105251.png"/>

> to use the cross-validation set to look at how many anomalies is finding, and also how many normal engines is incorrectly flagging as an anomaly.
>
> - based on this, try to choose a good choice for the $\epsilon$ parameter.

1. fit the model on training set.
2. On cv and test set, compute $p(x)$ for every example, and check how the result maches the labels $y$.
3. Apply (espicially for the skewed dataset) the $F_1$ score and evaluate the performance of the algorithm

So, build a anomaly detection algorithm is much easier if you have only a few **labeled examples of known anomalies**.

Here comes the question: why not take those labeled examples and use a supervised learning algorithm instead? 

Next:

>  let's take a look at a comparison between anomaly detection and supervised learning, and when you might prefer one over the other. 

## 3.5 Anomaly detection vs. supervised learning

The decision between using supervised and unsupervised learning is actually quite subtle in some applications. Here's some suggestions.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609110545.png"/>

- mainly decide on the dataset. 
- Why only 20 positive examples (small scale) lead to inappropriate use of supervised learning? 
  - 20 examples may not cover all the way that aircraft engines that will go wrong.
  - Then supervised learning algorithm **may not recognize** a new problem it never have seen (because it assumes future positive examples to be similar to those already exists)
  - But the **anomaly detection**, can recognize them.
- The way that those two algorithms look at the dataset is **different**.
- For example, **anamaly detection is often used in detecting financial fraud**. - because new fraud methods come up every month. 
  - **Supervised learning algorithm is used in detecting spam emails** - new spam emails may look similar to those previous ones. 

A few more examples:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609113104.png"/>

- **Anomaly detection**: finding new, *previously unseen* defects in manufacturing.

  - especially some security related applications. - hackers always finding brand new ways

- **Supervised learning**: finding kown, *previously seen* defects. 

  - such as finding new (centain type of) defect.
  - predicting the weather, because only a handful kinds of weathers can occur.

  > "it tries to decide if a future example is **similar** to the positive examples that you've already seen"

Next:

>  Let me share some practical tips on how to tune the features you feed to anomaly detection algorithm.

## 3.6 Chossing what features to use

For anomaly detection, because there is no target labels $y$ to give it information in what features to choose, it is **more important for us to decide what feature to use** and delete irrelavant features.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609114413.png"/>

1.  try to make sure the features you give it are **more or less Gaussian** (more Gaussian is better). 
   - you can plot a histogram of the feature.
   - If it is not Gaussian, try to make it more Gaussian. 
     - for example, make $(x_2 + c)$ instead of $x_2$ to move the $\mu$ of the feature.
     - or make $\log (x_2+c)$ to reshape the feature.
     - transform $x_3$ to $x_3^{\frac 1 2}$ ... etc.
   - A more Gaussian distribution of the data, makes the algorithm easier to fit and, when making prediction, identify those examples that are anomalous.

How this was implemented in Jupyter notebook:

- plot it originally:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609114549.png"/>

- transform to $x^{\frac 1 2}$:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609114610.png"/>

- to $x^{0.25}$:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609114647.png"/>

- and turn back to $x^{0.4}$:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609114809.png"/>

- That's it! And then try this form: $\log (x+c)$ 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609115134.png"/>

- adjust the value added:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609115212.png"/>

- "In ml literatures, there are some methods to automatically adjust the $x$, but practically dosen't make a lot difference."

- **Note**: whatever transformation you apply to the training set, please remember to **apply the same transformation** to your *cross validation* and *test* set data as well.

If the algorithm doesn't work that well on your cross validation set, you can also carry out an **error analysis process** for anomaly detection:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609121610.png"/>

- One common problem is finding $p(x)$ is too large for deciding a distinct threshold $\epsilon$ to divide the normal and anomaly.

- to look at this anomaly example, and figure out **what on earth makes it anomalous** even its $x_1$ feature looks pretty normal.

  - if I can identify some **new feature**, $x_2$, to help distinguish this example, then the algorithm is improved.

- For example, $x_2$ of this example is "insanely fast typing speed", then it tells this anomalous example apart.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609121550.png"/>

One more example, to monitor computers in a data center

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609122235.png"/>

- Here we firstly choose some features that might take on usually **very large values** when in the event of an anomaly.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609122006.png"/>

- But one occasion will be unusual if the computer takes **very high CPU load** and **very low network traffic**. So based on this, we make a new feature $x_5$, or even $x_6$:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240609122201.png"/>

- So that $p(x)$ is still large for the normal examples, but it becomes small in the anomalies. 

Next:

> Next week, we'll go on to talk about recommender systems. 
>
> When you go to a website and recommends products, or movies, or other things to you. How does that algorithm actually work?

# 4 Recommender System

## 4.1 Making recommendations

The commercial  impact and the actual number of practical use cases of **recommender systems** seems to be even vastly greater than the amount of attention it has received in academia.

- Amazon, Netflix...
- They recommend things to you and boost the commercial growth.

**Example**: predicting movie ratings.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240610123750.png"/>

- Your users rates movies using 1-5 stars (here we use a range of **0-5** stars).
  - "?" denotes no data for the rating
- You have some *users* and *items* that may be recommended to users.
  - $n_u$ denotes the number of users. Here $n_u=4$
  - $n_m$ denotes the number of movies. Here $n_m=5$
  - $r(i,j)$ is a boolean value to denote whether **movie i** has been rated by the **user j** . 
  - $y^{(i,j)}$ here is the rating of **user j** to the **movie i** 

How to do next:

- use the movies user had previously rated to predict **how this specific user would rate a new movie**
- So, we have confident in recommend what new movie to that user.

## 4.2 Using per-item features

Here, we temporarily add features $x_1$ and $x_2$ to the dataset, to tell us if the movie is *romance movie* or *action movie*.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240610171853.png"/>

- Later in this week we will see if these two features are removed, then how do we develop the algorithm,

  - now, we use $n=2$ to denote these two features.
  - the features for movie 1 (love at last) will be $x^{(1)}=\begin{bmatrix}0.9\\0\end{bmatrix}$

- Here we use (the same as linear regression) $w^{(j)}\cdot x^{(i)} + b^{(j)}$ to predict a user's interest for a particular movie. 

  - $w^{(j)}$ and $b^{(j)}$ are associated with user. For example, $w^{(1)}$ and $b^{(1)}$ are parameters for **user 1, Alice**.
  - $x^{(i)}$ is the features of **movie i**. 

- For example, to predict Alice's rating for "Cute puppies of love" movie, we get this result, means she would probably give this movie a 4.95 score, very high:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240610171751.png"/>

Now, let's look at the cost function of this model:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240610173212.png"/>

- Note a new notation, $m^{(j)}$, refers to number of moviews rated by user $j$.

- **Our objective is to learn parameters $w$ and $b$ for user $j$**.

- The cost function is almost the same as before in linear regression (Mean Squred Error) 

  - Note that for the "sum" part of this function, we only sum up the movies where $r(i,j)=1$ , because only them was actually rated.

- Moreover, you can still add a regularization term to this cost function. note the $n$ here refers to the number of features

  - for recommender system, eliminate the "division by $m^{(j)}$" term is more convenient.

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240610173149.png"/>

- If you minimize this cost function, you end up with pretty good value of $w^{(j)}$ and $b^{(j)}$, helping you make good predictions for user $j$.

Go further into cost function, to expand it for all users:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240610173943.png"/>

- instead of focusing on a single user, let's look at how we learn the parameters **for all of the users**

- So for now, we add a summation for an extra dimension (number of users) $\sum^{n_u}_{j=1}$ to the cost function:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240610173827.png"/>

- It indeed very similar to linear regression cost funtion, only for now we're **training a different linear regression model for every user**.

So that's how you can learn parameters and predict movie ratings, if you had access to these features $x_1$ and $x_2$, which tells us how much each of the movies are *romance* or *action* movies.

Next:

> we'll look at the modification of this algorithm. 
>
> They'll let you make predictions that you make recommendations, even if you don't have in advance, features that describe the items of the movies in sufficient detail, to run the algorithm that we just saw

## 4.3 Collaborative filtering algorithm

**what if you don't have those features, $x_1$ and $x_2$**? Let's look at how you can learn to come up with those features $x_1$ and $x_2$ from the data.

Now we have a same dataset except unknown values for $x_1$ and $x_2$:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240611223135.png"/>

- If we already learned parameters for these four users: 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240610174529.png"/>

  - **we'll look at how we can get them**. But now for the purpose of illustration, let's say, we have them already.
  
- for the model $w^{(j)}\cdot x^{(i)}+b^{(j)}$, If we eliminate the parameter $b^{(j)}$ at first. So the simplified model is like:
  $$
  prediction = w^{(j)}\cdot x^{(i)}
  $$

  - According to the given dataset:

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240611224119.png"/>

  - So now, the question is: **what value of $x^{(1)}$ would lead to these four dot product results to be right**?

  - To make a computation, only when $x^{(1)}$ being the vector of $\begin{bmatrix}1\\0\end{bmatrix}$ can make these dot product result to be **5, 5, 0 and 0**.

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240612174248.png"/>

    - (note that $x^{(1)}$ is **the feature vector of movie 1**, contains $x^{(1)}_1=1$ and $x^{(1)}_2=0$) 
    - this means: based on the four user's rating for movie "Loce at last" (the first movie), we can make a prediction of the feature vector $x^{(1)}$

  - Similarly, we can try to also come up with a feature vector $x^{(2)}$ for the second movie and $x^{(3)}$ for the thir movie and so on, to try to make the algorithm's predictions on these additional movies close to what was actually the ratings given by the users.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240612174438.png"/>

- in a typical linear regression application, if you had just a single user, you don't actually have enough information to figure out what would be the features, $x_1$ and $x_2$
  - here we can compute in a typical linear regression way is because we set parameters $w$ as already known
- But in collaborative filtering, is because you have ratings from **multiple users of the same movie**. That's what makes it possible to try to guess what are possible values for these features, only given parameters associated with all user's, $w^{(1)}, b^{(1)}$ to $w^{(n_u)}, b^{(n_u)}$

Here is the cost function of **learning the feature vector** $x^{(i)}$:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240612190210.png"/>

- To learn features for a single movie; to learn features for all of the movies in the dataset - all shown in ppt above

  - Given $w^{(j)}, b^{(j)}$, to learn one feature vector $x^{(i)}$ for a specific movie $i$:
    $$
    J(x^{(i)})=\frac 1 2 \sum_{j:r(i,j)=1} (w^{(j)}\cdot x^{(i)}+b^{(j)}-y^{(i,j)})^2+regularization_{x^{(i)}_k}
    $$
    To be specific, this slide is important:

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240612174438.png"/>

    - The overall error of, say "Love at last" movie (the **1st** movie) is calculated by the $J(x^{(1)})$ - this cost function sums up the loss of $\hat y$ predicted by given parameters for $w^{(1)},...,w^{(4)} $ and the (not optimized, raw value) feature vector $x^{(1)}$ .
    - Taking derivative for the $J(x^{(1)})$  (⚠️ this time, not about $w$ or $b$, but $x^{(1)}$), will tell us how the movement of $x^{(1)}$ will lead to the reduction in value of $J(x^{(1)})$.

    -  during every gradient, taking derivative for the $J(x^{(1)})$, gives us the direction of optimizing values of feature vector $x^{(1)}$. We implement this 👇 (similarly) to ensure the $x^{(1)}$ is going into the right way

      <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20230528225616.png"/>

      - please replace the $J(w,b)$ in the slide above to the $J(x^{(1)})$, with $w, b$ to the $x^{(1)}_1, x^{(1)}_2$ (to elements consists of feature vector $x^{(1)}$)

    - after several epochs of gradient descent, we get the final value of $x^{(i)}$ 

  - minimizing the cost functions will give you pretty good prediction of $x^{(i)}$.

- This is pretty remarkable for most machine learning applications, the features had to be externally given. But in this algorithm, **we can actually learn the features for a given movie**!

So, **what is the collaborative filtering algorithm**? 

- We put together algorithm to learn the $w$ and $b$ in the last video, and the algorithm to learn to feature vector together
- this gives us **the collaborative filtering algorithm**
- this algorithm enables us carry out learning process without knowing both feature vector $x^{(i)}$ and parameters $w^{(j)}, b^{(j)}$

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240613215555.png"/>

- this 👇 two summation terms actually do the same thing, that is, **summing over all of the pairs where the user had rated that movie**, although in two different ways.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240613215938.png"/>

- this is the overall cost function of learning $w, b$ and $x$ 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240613215629.png"/>

- It turns out that if you minimize this cost function as a function of $w$ and $b$, as well as $x$, **then this algorithm actually works**!

So, how do you minimize this cost function as a function of $w,b,x$ at the same time?

- you still use gradient descent. Here is what we use gradient descent in course 1:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240613220706.png"/>

- with collaborative filtering, you will need to **update $x$ as well** in each epochs.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240613220951.png"/>

- So, $x$ is also a parameter. So if you do this, then you actually find pretty good values of **w** and **b**, as well as **x**.

A little explanation of the name "collaborative filtering":

> the name collaborative filtering refers to the sense that because multiple users have rated the same movie collaboratively, given you a sense of what this movie maybe like. 
>
> That allows you to guess what are appropriate features for that movie, and this in turn allows you to predict how other users that haven't yet rated that same movie may decide to rate it in the future.

Next:

> So far, our problem formulation has used **movie ratings from 1- 5 stars or from 0- 5 stars**. 
>
> (yet) A very common use case of recommender systems is when you have binary labels such as that the user favors, or like, or interact with an item. 
>
> In the next video, let's take a look at a generalization of the model that you've seen so far to binary labels. 

## 4.4 Binary labels: favs, likes ad clicks

The process we'll use to generalize the algorithm will be very much reminiscent to how we have gone from linear regression to logistic regression, from **predicting numbers** to **predicting a binary label** back in course one.

Here's an example of a collaborative filtering data set with **binary labels**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240614122558.png"/>

- "1" denotes the used was engaged with the movie. For example, Alice may watched "Love at last" all the way to the end, but maybe after playing a few minutes of "nonstop car chases" decided to stop the video.
  - Or it could mean that she explicitly hit like or favorite on an app to indicate that she liked these movies.
- "?" may denotes that the user has not seen this movie yet.

There are many ways of defining what is the label "1" and what is the label "0", and what is the label "?" in collaborative filtering with binary labels:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240614122924.png"/>

Given these labels, let's see how we can generalize our algorithm from linear regression to binary classification:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240614142621.png"/>

- previously the function we were using was a lot like the linear regression model.

- Now, the function of binary classification is a lot like a logistic regression model:

  - Instead of:  $w^{(j)}\cdot x^{(i)}+b^{(j)}$

  - Now it is: $g(w^{(j)}\cdot x^{(i)}+b^{(j)})$

  - **Purpose**: predicting the probability of $y^{(i,j)}=1$ -> which means a specific user is engaged in this movie.

    > (previousely we directly predict the rating number of the movie by a specific user)

We also need to modify the cost function:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240614143557.png"/>

- The loss function is almost the same as binary cross entropy loss function:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240614143533.png"/>

  - this computes the loss for a single example

- So now, summing up all the losses for all examples, gives us the **cost function** of the binary classification problem:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240614143500.png"/>

That's how you can take the linear regression-like collaborative filtering algorithm and generalize it to work with binary labels!

And this actually very significantly opens up the set of applications you can address with this algorithm.

Next:

> there are also some implementational tips that will make your algorithm work much better. 
>
> Let's go on to the next video to take a look at some details of how you implement it and some little modifications that make the algorithm run much faster.

# 5 Recommender systems implementation

## 5.1 Mean normalization

In the case of building a recommended system with numbers wide such as movie ratings 1-5 or 0-5 stars, your algorithm will run more efficiently, and also perform a bit better if you first carry out **mean normalization**.

- that is, if you normalize the movie ratings to have a consistent average value

down below is the dataset we used before. Now we add a new user, **Eve**, who has not rated any movies before.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615104632.png"/>

- It turns out that if we don't use mean normalization (only use the function with regularization), then this new user Eve's rating to all movies will be predicted to **0** by the algorithm. That's not right and not helpful.

Including mean normalization will improve the algorithm's performance regarding new user hugely. That's how it works:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615110708.png"/>

1. Extract the dataset and compile it into a matrix (upper left).

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615110750.png"/>

   - like shown above

2. compute average rating for each movie (2.5, 2.5, 2, 2.25, 1.25). And gather them into a vector, called $\mu$

3. Take the ratings for each movie and subtract its corresponding $\mu_i$. This gives a new matrix (upper right).

4. When you are carring out prediction, remember to **add $\mu_i$ back to the linear function**, to get a final result.

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615105907.png"/>

5. So this time, Eve's rating for each movie, according to the algorithm, is $0 + \mu_i$, which is much more accurate than before (this time, the rating is equal to the mean rating of previous users for that movie).

Mean normalization makes algorithm runs a little faster and predicts more accurate, especially for new users.

By the way, normalize the columns of the matrix (average rating of a specific user) is also considerable

- especially when there's a brand-new movie, that no one has rated yet. But in this case you probably should not show this movie to a lot users.
- but in this case, we only normalize the rows. That's enough. And it's more useful in most cases.

## 5.2 Tensorflow implementation

The Tensorflow can not only be used to build neural networks. It can also be helpful in building other types of learning algorithms.

- Tensorflow can also automatically help you to calculate derivatives of the cost function.
- All you need to do is to write a few lines of code to specify the cost function. You don't even need to know calculus to run gradient descent in Tensorflow.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615150605.png"/>

- Sometimes, computing the $\frac \partial {\partial w}$ derivative term can be very complicated
- Tensorflow can help us with that. Let's see how.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615161236.png"/>

- In this case, we use a very simple cost function $J=(wx-1)^2$ , so we're not optimizing $b$.
- `w = tf.Variable(3.0)` tells the Tensorflow that `w` is the parameter we want to optimize.
- simply use `[dJdw] = tape.gradient(costJ, [w])` will do the derivative as well as gradient descent. The `[w]` specified in the brackets tells tensorflow we want to take partial derivatives of `w` in the cost function `costJ`
  - before this, you need to tell tensorflow how to compute `costJ` in this expression: `with tf.GradientTape():`
  - Then, tensorflow will automatically record the sequence of computing the `costJ` in the `tape` 
- At last of each iteration step of optimizing the parameter `w`, remember to update the tensorflow variable `w` by this expression: `w.assign_add(-alpha * dJdw)`

So you mainly need to tell tensorflow how to compute the cost function `costJ`.

- And the rest of the syntax causes TensorFlow to **automatically** figure out for you what is that derivative $\frac \partial {\partial w}J(w)$ 
- this is a very powerful feature of tensorflow called **Auto Diff**
  - sometimes people call this **Auto Grad**, but it's not correct
  - *Auto Grad* is actually the name of the specific software package for doing automatic differentiation
- PyTorch also supports this feature.

This is the syntax of implementing the collaborative filtering algorithm in Tensorflow:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615162600.png"/>

- Note that here, the optimizer is **Adam** with learning rate specified as `1e-1`
- The cost function is specified in this line of code: `cost_value = cofiCostFuncV(...)` - and packed in the `tape`
- Finally in each iteration loop, we apply Adam optimizer using `optimizer.apply_gradients(zip(grads, [x,W,b]))` to update the value of the variables (parameters).
  - `zip()` function in Python is to just rearrange the numbers into an appropriate ordering for the applied gradients function.

Next:

> I'd like to also move on to discuss more of the nuances of collateral filtering, and specifically the question of **how do you find related items, given one movie, whether other movies similar to this one**.

## 5.3 Finding related items

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615171113.png"/>

- it is hard to tell that features learned by collaborative learning algorithm means what.
- finding **item k** with a feature similar to $x^{(i)}$ will give us the similar movie
  - Use squred distance $||x^{(k)}-x^{(i)}||^2$ to measure how similar two movies are
  - you can find 5-10 closest movies k

Some limitations of Collaborative filtering:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615171646.png"/>

1. it doesn't work good at cold start problems
2. It doesn't give you a natural way to use side information or **additional information** about items or users.
   - for examples, it is not good at using a user's location / age / gender to more accurately predictions of his or her preferred movie.
   - even these information are acturally very helpful

 Next:

> let's go on to develop content-based filtering algorithms, which can address a lot of these limitations. 
>
> - Content-based filtering algorithms are a state of the art technique used in many commercial applications today.

## Programming Assignment

This slide clearly explains what the collaborative leaning algorithm is doing:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615205434.png"/>

- The goal of a collaborative filtering recommender system is to generate two vectors: 

  1. For each user, a 'parameter vector' $w^{(j)}$ that embodies the movie tastes of a user. 
  2. For each movie, a feature vector $x^{(i)}$ of the same size which embodies some description of the movie.

  The dot product of the two vectors plus the bias term should produce an estimate of the rating the user might give to that movie.

  - $Y(i,j) = R(i,j)(w^{(j)}\cdot x^{(i)}+b)$

- One training example is shown above: $\mathbf{w}^{(1)} \cdot \mathbf{x}^{(1)} + b^{(1)} = 4$

- Use this 👇 to check the notations

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240615211315.png"/>

# 6 Content-based filtering

## 6.1 Collaborative filtering vs. Content-based filtering

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616112958.png"/>

- collaborative filtering algorithm only take in mass rating numbers associated with users && items to try to based on a user's former ratings (and ratings distribution of other movies by other users), figure out what this user would like.

  > "Recommend items to you based on ratings of users who gave similar ratings as you"

- Compared with collaborative filtering algorithm, the **Content-based filtering** algorithm looks at the features for both user and the item, and try to match them.

  - thus it is capable of predicting a new user's prefereed item based on this user's backgroud information (and the content base of the system).

- To study the Content-based filtering, we continue to use:

  - $r(i,j)$ to denote wehther user $j$ has rated item $i$
  - $y(i,j)$ to denote the user $j$ 's rating to item $i$

Here is the example users && item features of a typical content-based filtering algorithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616122252.png"/>

- Some explaination for the features
  - **User-country**: 1-hot feature with around 200 possible values.
  - **User-movies watched**: construct 1000 features that tells you the 1000 most popular movies in the world and which of these the user has watched
  - **User-Average rating per genre**: the average rating for this user regarding "romance movies", "action movies" and so on
- you can construct two feature vectors:
  - $x_u^{(j)}$ for user $j$
  -  $x_m^{(i)}$ for movie $i$
- The task is to try to figure out whether a given movie $i$ is going to be good match for user $j$.

The function of content-based filtering:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616123453.png"/>

- for content-based filtering, we won't use $b$ as a parameter. Instead:
  - $v_u^{(j)}$ stands for the vector computed from the features of user $j$
  - $v_m^{(j)}$ is computed from movie $i$
- The dot product of these two vectors, hopefully give us a sense of **how much this particular user will like this particular movie**.
  - like, the $v_u$ denotes the user's preference, and the $v_m$ denotes several features of a given movie.
- Now, the question is how can we compute the $v_u^{(j)}$ and $v_m^{(j)}$
- Notice that $x_u$ and $x_m$ can be different in size, but the $v_u$ and $v_m$ cannot, because these two need to dot product with each other.

## 6.2 Deep learning for content-based filtering

A good way to develop a content-based filtering algorithm is to use **deep learning**. The approach we will introduce next is the way that many important commercial state-of-the-art content-based filtering algorithms are built today.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616124735.png"/>

- To compute $v_u$, we are going to use a neural network
  - here because the nerons in the output layer, the $v_u$ is a list of 32 numbers
- Similarly for computing the $v_m$.
- these two neural networks can have different number of neurons in hidden layers. **But the size of output layer must be the same**.
- Instead of simply output the $v_u \cdot v_m$, here we can apply a sigmoid function to it $g(v_u\cdot v_m)$ , namely prediction of user $j$ on movie $i$, $y^{(i,j)}=1$ 

We can tie these two neural networks together to get the workflow:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616142300.png"/>

- this model has a lot of parameters. How can we train both parameters for user and movie network?

- The solution is to construct a cost function $J$, which is very similar to the cost function in collaborative filtering.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616140642.png"/>

  there's no separate training procedure for the user and movie networks. This expression is the cost function used to train **all the parameters** of the user and the movie networks.

  - If you want to regularize the model, add a neural network regularization term to encourage the neural networks to keep the parameters small.

This is akin to what we have seen with collaborative filtering features, helping you find similar items as well:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616141310.png"/>

- The distance between $v_m^{(k)}$ and $v_m^{(i)}$ describe how similar the movies are.
  - this is similar to the expression in collaborative filtering previously
- This can be **pre-computed**, means you can store the $v_m$ of movies, and recommend to a user once a specific movie is watched.

> one of the benefits of neural networks is that it's easier to take multiple neural networks and put them together to make them work in concert to build a larger system. This is the case.

One note of implementing this:

- works with finding good features, is almost the most important step in the commercial practices.

One limitation of this application:

- it can be computationally very expensive to run if you have a large catalog of a lot of different movies you may want to recommend.

Next:

> let's take a look at some of the practical issues and how you can modify this algorithm to make it scale that are working on even very large item catalogs.

## 6.3 Recommending from a larger catalogue

Today's recommender systems will sometimes need to pick a handful of items to recommend, from a catalog of thousands or millions or 10s of millions or even more items.

- To run a neural network inference every time when a new user shows up on your website, becomes comtutaionally infeasible.

- So, many large recommendation systems are implemented as two steps: **Retrieval and Ranking**

- In the **retrieval step**, you get many hundreds of different movies. And the purpose of this step is to ensure that the result covers enough good movies that the user may like

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616143719.png"/>

- In the ranking step (for a particular user), you take all the retrieved movies, and use the learned model to rank them.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616144236.png"/>

  - You need to only do this part of the neural network inference one time to compute $v_u$:

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616144035.png"/>

  - And take the inner product with the each $v_m$ computed for each potential plausible item. To get the score for the ranking.

- How many items you should collect for retrieval step:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616144653.png"/>

  - do offline experiments to see if retrieving additional items results in better recommendations, which means the estimated probability of $y^{(i,j)}=1$ is higher.

Next:

> So as you build your own recommender system, I hope you take an ethical approach and use it to serve your users. And society as large as well as yourself and the company that you might be working for. 
>
> Let's take a look at the ethical issues associated with recommender systems.

## 6.4 Ethical use of recommender systems

I hope you only do things that make society at large and people better off. 

Let's take a look at some of the problematic use cases of recommender systems, as well as ameliorations to reduce harm or to increase the amount of good that they can do. 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616145652.png"/>

- The goal of the recommender system can affect the real world hugely.
  - the recommender systems 3-5 may be problematic use cases.

Let's look at more details:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616150123.png"/>

- Travel industry, is a good example
- Payday loans is not. Because the principle of the business is unethical.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616150508.png"/>

- conspiracy theories / violent / fraud are naturally more engaging..
- many users don't realize that many apps and websites are trying to maximize their profit rather than necessarily the user's enjoyment of the media items that are being recommended

As a very lucrative, profitable technology, recommender systems (and other machine learning technoques as well) can also be harmful in some cases. **Please only build things and do things that you really believe can be society better off**. 

## 6.5 Tensorflow implementation

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616151928.png"/>

1. Construct two neural networks in a traditional way, use `activation = 'relu'`

2. tell the tensorflow how to feed the user features / item features into the two neural networks.

   Use user's end as an example:

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616151427.png"/>

   1. take the `input_user`
   2. feed to the `user_NN` to compute the `vu` 
   3. normalize the `vu` to have length **1**

3. take the dot product of the two outputs:

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616151615.png"/>

   - This is a special `keras` layer that only takes dot produts of two vectors.
   - this gives us the final prediction

4. compile all these together, specify the inputs and outputs of the model; And specify the cost function (MSE cost function)

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616151904.png"/>

