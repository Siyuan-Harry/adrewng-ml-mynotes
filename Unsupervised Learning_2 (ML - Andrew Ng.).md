# 7 Principal Component Analysis

## 7.1 Reducing the number of features

PCA is commonly used for visualization. Specifically, if you have a dataset with a lot of features, say 10 features, 50 features or even 1000 features, you can't plot 1000 dimensional data.

- PCA, or principal components analysis, can reduce the number of features to **two features**, or **three features**.
- So that you can plot it and visualize it,
- So data scientists can see what is going on.

To exemplify:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616202800.png"/>

- here, $x_1$ feature denotes the **length** of the car, and $x_2$ denotes the **width** of the car. 
  - **Note**: the width of the car is generally constrained a lot because of the width of the road.
- $x_1$ varies a lot but $x_2$ (width) almost show no difference from car to car.
- PCA will determine that, let's  just use $x_1$ feature. 

Another example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616202902.png"/>

- here again, the PCA will decide to just take $x_1$ as a valid feature.

A more complex example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616203528.png"/>

- It seems two features (length && height) are linear correlated. Some cars are bigger, that they **have larger length and larger height at the same time**

- So we construct a **z axis**, which includes both information for $x_1$ and $x_2$ (a linear combination of them both!)

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616203551.png"/>

  - **z axis** can denote the **size** of a car.

- So here is what PCA is doing: find new axis and coordinates. 

One more example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616203749.png"/>

- almost all data points in 3D graphic are living in a very thin surface:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616203729.png"/>

  - almost like a pancake

- So the PCA can do is, just reduce one dimension (by making combination of dimensions in 3D graphic)

  - "show the pancake in 2D surface"

- The PCA here, creates $z_1$ and $z_2$, two new dimensions.

Next example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616204208.png"/>

- If there are 50 features to measure a country. PCA can compress them into 2 features

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616204312.png"/>

- In the end, you may notice that the two created features $z_1$ and $z_2$ may correspondes to some original features..

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240616204602.png"/>

  - Such as "country's GDP", "per person GDP"

PCA is espicially powerful in visualizing the dataset. And carrying out data visualization is always helpful in letting us understand deeper about the data.

Next:

> let's start to take a look at how exactly the PCA algorithm works.

## 7.2 PCA Algorithm

How to use one feature, that is more effective, to represent the data originally have 2 features?

- Here we have $x_1$ and $x_2$, and there is no label $y$ 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618120525.png"/>

- You have to **preprocess the features** before utilize the PCA Algorithm, such as

  - Normalize the mean to be 0
  - Do the feature scaling (like the course 1, week 2, housing example)

- **Not bad approach**: project each point down to the z axis.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618120738.png"/>

  - for this example 👆 is not bad, because the variance of the data points in z axis is still large, so we still have a lot of information of original data after the projection.

- **Bad choice of the z axis**: the projected data points are squashed together

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618120945.png"/>

- The **best choice** of the z axis:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618121154.png"/>

  - In PCA algorithm, this axis is called, **the principal component** (first principle component)
  - which means, if you project all the data points onto it, you get the **largest variance**

> The scikit-learn can help you implement this algorithm.

Here are more details about this PCA algorithm:

- How to project a data point $(2,3)$ to the z axis?

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618122431.png"/>

  - Suppose the **length 1 vector** in z axis is  $\begin{bmatrix} 0.71\\0.71 \end{bmatrix}$ 

  - You take dot product of the data point to the z axis' length 1 vector:
    $$
    \begin{bmatrix} 2\\3 \end{bmatrix} \cdot \begin{bmatrix} 0.71\\0.71 \end{bmatrix} = 3.55
    $$

  - This 3.55 is the projected value of data point $(2,3)$ onto the z axis.

- If you were to pick another "2nd principal component", this axis will always be perpendicular to the 1st axis you picked. 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618122709.png"/>

  - If there will be the 3rd one, this one is also perpendicular to previous two.

How PCA is different from linear regression?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618123436.png"/>

- **In linear regression**, there are $(x,y)$ pairs, means there exists label $y$ . Its mission is to predict values in the fitted line as close as possible to the ground truth label $y$. Meaning that it is trying to minimize the distance (the **cost**) along $y$ axis.
- **In PCA algorithm**, you are try to reduce the distance between the unlabeled data points to the z axis, to retain variance as large as possible.
- If you have more than 2 features (which is almost all the case), the difference between two algorithm becomes very large

here is an extreme example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618123649.png"/>

- the line that PCA choose is totally different from the linear regression.

Given projected number $z=3.55$, can we reconstruct original coordinates of the data point?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618124018.png"/>

- we don't have enough info to get it back precisely
- But we can get it back approximately, which is $\begin {bmatrix} 2.52 \\ 2.52 \end {bmatrix}$
- That's not a bad approximation! This denotes the PCA retains a lot information of this data point during projection.

## 7.3 PCA in code

>  How you can use the scikit-learn library to implement PCA

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618134031.png"/>

1. **pre-processing**: feature scaling would be important to help PCA find a good choice of axes for you.
2. call the "`fit`" function to obtain new axes. The mean normalization is included in this step
   - as shown before, these axes (principle components) are perpendicular with each other.
3. Examine how good these axes are, by calling `explained_variance_ratio`.
4. Transform (project) the data onto new axes. 

Example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618134223.png"/>

1. we are to reduce features of `X` to one

2. that's the array on the left

3. The "explained_variance_ratio" equals 0.992, means that the principle component axis we founded can explain the 99.2% of the variance of the original data.

4. finally, call `inverse_transform` to reconstruct the examples into 2D graph again. We can see the result, now they lies into a single line.

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618134300.png"/>

By the way, we can also transform the original data, to 2D still:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618135138.png"/>

- The two new axes $z_1$ explains 99.2% of the variance of the original data and $z_2$ explains 0.8%. 
  - together they explains 100%
- so if you firstly apply this "2-axes" PCA fitting to the dataset, you may then discover that the **axis 1** retains most of the information of the original dataset. Then you can decide just to use this axis to represent the data.

More advice in applying PCA:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618135702.png"/>

1. They are useful in visualizing the data, even for now.
2. Because the modern storage space & transmitting techniques, the PCA used for data compression is less than before.
3. Because the modern deep learning (NN structure), the PCA also is less used for boosting the training speed as it does not help much.

# 8 Reinforcement learning introduction

In machine learning, reinforcement learning is one of those ideas that while not very widely applied in commercial applications yet today, is one of the pillars of machine learning. And has lots of exciting research backing it up and improving it every single day. 

## 8.1 What is Reinforcement Learning?

What is reinforcement learning:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618155145.png"/>

- How to write a programme to fly this helicopter? (compared with quad roter drones, radio controlled helicopters are harder to keep balance in the air)

  - reinforcement learning has been used to get helicopters to fly a wide range of stunts or we call them **aerobatic maneuvers**.

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618155313.png"/>

So how do you write a program to do this automatically?

- the task is to find a function, which mappes from the state $s$ to the action $a$

- supervised learning is not recommend, and it is not helpful here

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618155757.png"/>

  - because it is very ambiguous that the action to take (label y) corresponding to the each state (training example x). 

- we instead use **reinforcement learning**

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618160750.png"/>

  - the core of the reinforcement learning is to define a **reward function** 

  - it's like training a puppy dog using behaviorism approach.

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618160630.png"/>

  - The job for reinforcement learning is to get more good behaviors for a helicopter and less bad behaviours.

  - in reinforcement learning, you just tell it "what to do" rather than "how to do it". 

    > Let it learn the "secrets & inner techniques" by itself.

  - specifying the **reward function** rather than the optimal action gives you a lot more flexibility in how you design the system.

  - for example, it the helicopter flys well, you give it "+1" reward every second. But if it crashes, you give it "-1000" reward.

Today, reinforcement learning has been successfully applied to a variety of applications:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618161008.png"/>

- **Robot controlling**: in fact later this week in the practice lab, you implement for yourself a reinforcement learning algorithm to land a lunar lander in simulation.

the key idea of reinforcement learning is:

- rather than you needing to tell the algorithm what is the right output y for every single input, 
- all you have to do instead is specify a reward function that tells it when it's doing well and when it's doing poorly.

## 8.2 Mars rover example

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240618162616.png"/>

- The mars rover has 6 states, and not it starts at **state 4**

- The state 1 is the scenario that it encounters some highly scientific valuable surfaces on the Mars. The state 6 is valuable too, but not as much as state 1

  - the reward at state 1 is 100
  - the reward at state 6 is 40
  - Others are 0, because there are not that much interesting science works to be done in them

- The state 1 and 6 are "terminal state" because once get there, the mars rover ends its work for a day.

  - it can either choose to go left and go right.

- At every timestep, the robot is in some states, get to choose some actions, and it also enjoys some rewards $R(s)$. And it also gets to some new states $s^`$ (s prime)
  $$
  (s,a,R(s),s^`)
  $$

- That's the formalism of how a reinforcement learning application works

## 8.3 The Return in the reinforcement learning

But how do you know if a particular set of **rewards** is better or worse than a different set of rewards? The "**Return**" in reinforcement learning allows you to capture that.

- The concept of a return captures that **rewards you can get quicker** are maybe more attractive than rewards that take you a long time to get to

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240619151029.png"/>

- In this mars rover example, if you go left, you finaly get a 100 reward. But during each of the timestep, you must multiply the **discount factor** $\gamma$ with the award corresponding to that status

  - here, the discount factor is (0.9) and $0.9^2, 0.9^3$, according to how far the mars rover already gone
  - So the return of go left becomes **72.9**
  - Remember **the first reward is not discounted**

- The return is the reward multiply the discount factor in each step, and summarize them. 

- This gives the algorithm the ability to evaluate between different directions to go (based on the combination of distance and reward) , and pick one with the highest reward.

- If $\gamma = 0.5$, the return to go left is close to the go right, even go left will end up being the status with 100 reward.

  12.5 (left), compared with 10 (right)

In financial applications, the discount factor can be very naturally interpreted as "interest rate" or "time value of money".

Some more examples on "Return":

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240619152006.png"/>

- The return depends on the action you take and the start point
  - if you start at status 4, then choose to go left, then you end up with 12.5 return
  - if you start at status 2, then the return is 50
- If you are to go right, things are different.
- Finally, after learning all the returns for every direction from every starting point, you decide that go right when start at **status 5**, instead for **status 1-4**, go left.

Next:

> To formalize the goal of reinforcement learning

## 8.4 Making decisions: Policies in reinforcement learning

There are many different strategies that the algorithm can choose:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240619175051.png"/>

- like:
  1. always go for the nearer reward (leftmost / rightmost)
  2. Go for higher reward
  3. go for smaller reward
  4. choose to go left unless one step away from the lesser reward

- we need a function $\pi$ that mappes the **action** $a$ to the current **status** $s$
  - $\pi(5) = left$ means choose to go left when status is 5

To goal for the reinforcement learning:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240619175250.png"/>

- call $\pi$ as "policy" is a converntional way, though it is not that natural
- Maybe understand the "policy" as "controller" is better

## 8.5 Review of key concepts

Let's do a quick review of the key concepts and also see how this set of concepts can be used for other applications as well:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240619175948.png"/>

- In helicopter example:
  - **states** can be a set of positions of helicopter
  - **actions** is the set of possible ways to move the controls stick
  - the **policy** is to find the proper movement $a$ regarding state $s$ 
- In chess examples:
  - **states** is the positions of all the pieces on the board
  - **reward** is +1 if wins a game, 0 if ties a game
  - the goal is given a board position, to find a good action using **policy** $\pi$

This formalism of a reinforcement learning application actually has a name, **MDP**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240619180258.png"/>

- 👆 this is how this process works
- "**The future only depends on the current state**"

Next:

> we'll start to develop an algorithm for picking good actions. 
>
> The first step toward that will be to define and then eventually learn to compute the state action value function. 
>
> This turns out to be one of the key quantities for when we want to develop a learning algorithm.

# 9 State-action value function

## 9.1 State-action value function defination

State-action value function is a key quantity that reinforcement learning algorithm will try to compute.

- This function is typically denoted by the $Q$ 

- This is the main explanation  of this function:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620101825.png"/>

  - Later, we will find a way to compute the state-action value function

Let's bring the mars rover example back. Here is the optimal policy that the rover moves:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620102043.png"/>

Let's figure out the $Q$ for each state:

- $Q(2, right) = 0+(0.5)\times 0 + (0.5)^2 \times 0 + (0.5)^3 \times 100 = 12.5$
  - Start from state 2 and go right, the trail is: 2 >>> 3 >>> 2 >>> 1
  - note that even go right from state 2 is not a good choice, $Q$ does not judge it right or wrong. But just compute the $Q$ here

- $Q(2, left) = 0+0.5\times 100 = 50$

- $Q(4, left) = 0+(0.5)\times 0 + (0.5)^2 \times 0 + (0.5)^3 \times 100 = 12.5$

- This is the Q-function for each state of this example:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620102744.png"/>

The best possible return / action from the state $s$ is the largest value of $Q(s,a)$ 👇

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620103408.png"/>

- For every state and for every action when you're in some state $s$, all you have to do is look at the different actions $a$

- You should pick the action $a$ That maximizes $Q(s,a)$, and so $\pi(s)$. 

- This turned to be the optimal action (maximizes the return at this circumstance)

- Sometimes at Internet, someone refer to "$Q^*$" and call this optimal $Q$ function. This is exactly same as the state-action value function in this class

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620103834.png"/>

Next:

> Let's look at some examples that tells us what these values look like

## 9.2 State-action value function example

In order to keep holding our intuition about reinforcement learning problems and how the values of $Q(s,a)$ change depending on the problem will provided an optional lab. That lets you play around modify the mars rover example and see for yourself how $Q(s,a)$ will change.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620104419.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620104458.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620104557.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620104628.png"/>

Next:

> After you play to the lab, we then be ready to come back and talk about what's probably the single most important equation in reinforcement learning, which is something called the **bellman equation**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620104935.png"/>

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620105017.png"/>

## 9.3 Bellman Equation

So the question is, how do you compute $Q(s,a)$? The Bellman Equation.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620110501.png"/>

- $s,a$ is the start state and action, $s^\prime, a^\prime$ is another state and action

- $$
  Q(s,a) = R(s) + \gamma \max_{a^\prime} Q(s^\prime, a^\prime)
  $$

- 向右：0 + (0.25)*0 + (0.25^2)*0 + (0.25^3)*40 = 0.625
- 向左：0 + (0.25)0 + (0.25^2)0 + (0.25^3)0 + (0.25^4)100 = 0.391

To varify this equation:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620110749.png"/>

And to breakdown this equation:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620111422.png"/>

- The bellman equation is cosisted of **two parts**:
  - the reward you get right away (multiplied by $\gamma^0$, which is 1), and
  - the best return from behaving optimally starting from $s^{\prime}$
- **It is the return if you start from state $s$, take action $a$ (once), then behave optimally after that.** 
- We can double check this breakdown in the example we calculated before:
  - <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620111713.png"/>

## 9.4 Random (stocastic) environment

In practice, many robots don't always manage to do exactly what you tell them because of wind blowing and off course and the wheel slipping or something else.

There's a generalization of the reinforcement learning framework we've talked about so far, which **models random or stochastic environments**. 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620112132.png"/>

- what if 10% of the time, the mars rover actually ends up accidentally slipping and going in the opposite direction? 
  - 0.9 chance to go for the right direction
  - 0.1 chance to go wrong

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620112409.png"/>

So here, we use "**Expected Return**" to denote an average from (maybe *many thousands of*) different returns of movement.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240620112646.png"/>

- So the job of reinforcement learning algorithm is choose a policy $\pi$ in which action $a$ maximizes the expected return 👇

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240621091904.png"/>

- now, the Bellman queation consisits of two parts:

  1. the reward you get right away (also called immediate reward), and
  2. what you **expect** to get **on average** of the **future returns**.

Now, back into the optional lab, set the misstep probability to **0.9**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240621134352.png"/>

- You can see the Q-values (optimal returns) have gone done a bit. Because you cannot control the robot as well as before.

# 10 Countinuous state spaces

## 10.1 Examples of continuous state space applications

Robots can be in continuous positions (cannot breakdown into concrete numbers)

- Such as the mars rover can be anywhere on a line

- for another example, when you want to control a toy truck, you have to clearify the **x-position**, the **y-position**, the **speed in x-y-position** and the **orientation** (where it faces) with the **angular velocity** of it.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240621191020.png"/>

  - so the state is an vector of six numbers

Another example: controling the helicopter

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240621191544.png"/>

- **Capturing the position of the helicopter**: the x-y-z axis
  - $[x,y,z]$
- **Capturing the gesture of the helicopter**: the roll, the pitch, the yaw
  - $[\phi, \theta, \omega]$
- **Capturing the speed of the helicopter**: the speed for x-y-z and angular velocity of its gesture
  - $[\dot x, \dot y,\dot z, \dot \phi, \dot \theta, \dot \omega]$
- The policy is to look at these 12 numbers and to decide what direction to go.

## 10.2 Lunar lander

The lunar lander lets you land a simulated vehicle on the moon. It's like a fun little video game that's been used by a lot of reinforcement learning researchers.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240621193057.png"/>

- You have to choose action(s) to ensure the successful landing

- The state is consists of these numbers:
  $$
  s=\begin{bmatrix} x\\y\\\dot x\\\dot y\\\theta\\\dot theta \\l \\r \end{bmatrix}
  $$

  - **x, y** refers to the position of lunar lander in the 2-D coordinate system
    - $\dot x, \dot y$ refers to the speed in both axes
  - $\theta$ refers to the angle of the lunar lander
  - $l,r$ are two binary numbers, which corresponds to whether the left / right leg is touching the ground

The reward function is as follow:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240621193839.png"/>

- You find when you're building your own reinforcement learning application usually takes some thought to specify exactly what you want or don't want and to codify that in the reward function.
- But specify the reward function should still turn out to be much easier to specify the exact right action to take from every single state.
  - this approach (specify the reward function rather than pointing out the exact right direction) is similar to the thought of "cost function" rather than find the minimum value mathematically.
  - "taking little step at a time" rather than "straightforwardly go for the global minimum"

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240621193947.png"/>

- The gamma can be 0.985
- And if you can learn a policy pi that does this then you successfully land this lunar lander 

we're now finally ready to develop a learning algorithm which will turn out to use **deep learning (neural networks)** to come up with a policy to land the lunar lander.

## 10.3 Learning the state-value function

The heart of the learning algorithm is we're going to train a neural network that inputs the current state and the current action, and computes or approximates $Q(s,a)$

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622093134.png"/>

- The input $\vec x$ : 12 number-vector
  - **8 numbers**: to decribe the state
  - **4 numbers** (binary for each): to decribe the action
- The job for this neural network is to output the $Q(s,a)$
  - also, target value $y$

What is different from supervised learning:

- Instead of input a state $s$, then try to output an action $a$,
- **The reinforcement learning algorithm is to input a state-action pair (the 12-number vector), and have it try to output the state-action value function $Q(s,a)$**

- So, whenever your lunar lander is in a state $s$, you can use the trained neural network to compute $Q(s,a)$ for all four actions -- then finally **you just pick whichever values higest**

how do you train a neural network to output $Q(s,a)$?

- to use Bellman's equations to create a training set with lots of examples, x and y.
- and then we'll use supervised learning, to learn a mapping from $x$ to $y$.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622095157.png"/>

- Firstly, we must clarify what are the training examples and targe values in this neural network:
  - **Training examples**: the $s$ and $a$
  - **Target values**: the $R(s) + \gamma \max_{a^{\prime}} Q(s^{\prime}, a^{\prime})$

- How can we generate the training examples and labels:

  - we randomly trying out different actions (maybe good or bad) in the lunar lander simulator.

  - And record the state $s^{(1)}$, the action $a^{(1)}$, the reward $R(s^{(1)})$ and the next state $s^{\prime (1)}$ 

  - Maybe you try this over 10,000 times. You get all these datas.

  - And you use $s^{(1)}, a^{(1)}$ to compute each training example $x^{(i)}$; Use $R(s^{(1)}), s^{\prime (1)}$ to compute $y^{(i)}$

- Another thing we need to clearify: initially we don't know the Q-function, so how can we compute $y$?
  - you can start off with taking a totally random guess of it.
  - The algorithm will work nonetheless. But in every step **Q** is just going to be some guess. 
  - They'll get better over time. it turns out of what is the actual Q function

Let's see what is the full learning algrithm:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622095943.png"/>

- The $Q(s^{\prime}, a^{\prime})$ is randomly intialized by this neural network.
- we train the neural network to come up with $Q_{new}$, and this $Q_{new}$ can approximately  mapping $x$ to $y$ .
- Finally you update Q to the $Q_{new}$. That's one training loop. If you train this NN for more times, and update $Q$ each time, then the result would be better and better.

The algorithm you just saw is sometimes called the **DQN algorithm** which stands for Deep Q-Network. Because you're using deep learning and neural network to train a model to learn the Q functions.

Next:

>  if you use the algorithm we just described, it will just sort of work. But it turns out that with a couple of refinements to the algorithm, it can work much better.

## 10.4 Algorithm refinement: Improved neural network structure

This is the nerual network archetecture we saw previously:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622110347.png"/>

- This is inefficient because we have to carry out inference **4 times** from every single state.
  - because there are 4 different actions can take.

So, the modified NN architecture looks like this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622110715.png"/>

- Output four $Q(s,a)$ simultaneously. 
- This utilizes the most of the modern GPU computation
- now, given a state $s$, we can run inference just once, and get all four values, to pick an action $a$ that maximizes $Q(s,a)$ 
- this also boosts the computation of $\max_{a^{\prime}} Q(s^{\prime}, a^{\prime})$

We will use this architecture in the practice lab.

Next:

> there's one other idea that'll help the algorithm a lot which is something called an Epsilon-greedy policy, which affects how you choose actions even while you're still learning. 

## 10.5 Algorithm refinement: $\epsilon$-greedy policy

The learning algorithm that we developed, even while you're still learning how to approximate $Q(s,a)$, you need to take some actions in the lunar lander. How do you pick those actions while you're still learning?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622111227.png"/>

- the most common way: $\epsilon$-greedy policy

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622112630.png"/>

- Option 1 is good, but not the best
- Option 2 is what we typically use in practice

Why option 2 is better:

- sometimes because of the random initialization, the $Q(s,main)$ is always low, means the return of firing the main engine is always low.
- In this situation, **if we use option 1**, then the algorithm will never ever try to fire the main engine, thus will never learn it. 
  - So that the algorithm will never have a chance to figure out that "firing the main engine sometimes can be a good idea!"
- **In Option 2**, this allows the neural network can learn to overcome its own possible preconceptions about what might be a bad idea that turns out not to be the case.

**Option 2** has the name: $\epsilon$-greedy policy

- In this example, the 0.95 is called "Greedy" or "Exploitation" 

  > let's exploit everything we've learned to do the best we can.

- the $\epsilon$ specifies the possibility (0.05) in which we explore a little bit of what we haven't done before.

The common trick is to start off $\epsilon$ high (initially take almost random actions) and gradually decrease it.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622112652.png"/>

- So that over time, you will start to less act randomly and using the experience you learned to act.

What's more:

1. After applying the more efficient NN architecture and $\epsilon$-greedy policy, the algorithm works pretty well on the lunar lander.
2. Reinforcement learning algorithms are much more finicky to little choices of parameters, compared with supervised learning

Next:

> to drive a couple more algorithm refinements, mini batching, and also using soft updates

## 10.6 Algorithm refinement: Mini-batch and soft update

The first idea is called using mini-batches, and this turns out to be an idea they can both speedup your reinforcement learning algorithm and it's also applicable to supervised learning.

The mini-batches:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622121541.png"/>

- this is the exact housing price example in course 1 (supervised learning)
- Sometimes, the training set size $m$ can be 100,000,000, in such circumstance, the each computation step of gradient descent will be too slow. (because will need to scan over all the 100,000,000 examples time after time)
- **The mini-batch**: take 1000 examples as $m^\prime$ 

For each iteration, the algorithm only look at one mini-batch. 

- For example, one way is in iteration 1, look at mini-batch 1
- in iteration 2, look at mini-batch 2
- ....

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622121900.png"/>

The difference in loss:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622122302.png"/>

- On average, the mini-batch gradient descent will tend to go onto the global minimum. 
- Even its trail looks like a little confused (because the ramdomly choosed mini-batches), it is totally bearable compared with the benefits that the computational resources it saved and the accelaration it brings about

This brings you a much faster learning algorithm especially when you have a very large training set.

In reinforcement learning, you might choose only a subset of the total 10,000 training examples (use only **1,000** at a time rather than 10,000).

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622122639.png"/>

- This will overall speed up the learning algorithm.

And, **soft updates** can make the algorithm coverge more reliably:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622123213.png"/>

- pay attention to the **"set $Q=Q_{new}$"** step.

- To prevent $W_{new}, B_{new}$ is worse than previous $W, B$

- In each step of updating the $W,B$, we set:
  $$
  W=0.01\times W_{new} + 0.99\times W \\
  B=0.01\times B_{new} + 0.99\times B
  $$

- This is called "soft update" - make more gradual changes.

## 10.7 The state of the reinforcement learning

> what I hope to do is share with you a practical sense of where reinforcement learning is today in terms of its utility for applications

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240622123729.png"/>

- Succeed in simulation is not a guarantee on the succeed in the real world.
- The potential for reinforcement learning for future applications is very large

## Practice lab

在强化学习中，尤其是使用深度Q网络（DQN）时，创建两个神经网络（Q-Network和Target Q-Network）的主要原因是为了提高训练的稳定性。这两个网络的作用和区别可以通过以下几点来解释：

1. **目标值的计算：**
   在Q-Learning中，我们需要计算目标值（target value），即：
   \[
   y = R + \gamma \max_{a'}Q(s',a';w)
   \]
   这里，\(R\) 是即时奖励，\(\gamma\) 是折扣因子，\(Q(s',a';w)\) 是下一状态的动作价值函数。

2. **目标值的变化：**
   如果直接使用Q-Network来计算目标值，那么目标值 \(y\) 会在每次迭代中不断变化，因为Q-Network的权重 \(w\) 在每次迭代中都会更新。这种不断变化的目标值会导致训练过程中的不稳定和震荡，难以收敛。

3. **引入Target Q-Network：**
   为了解决上述问题，我们引入了Target Q-Network（目标Q网络），其结构与Q-Network相同，但权重 \(w^-\) 更新得更慢。使用Target Q-Network计算目标值：
   \[
   y = R + \gamma \max_{a'}\hat{Q}(s',a';w^-)
   \]
   这里，\(\hat{Q}\) 是Target Q-Network，\(w^-\) 是其权重。

4. **权重更新：**
   每隔固定的时间步（通常是C步），我们将Q-Network的权重 \(w\) 复制到Target Q-Network的权重 \(w^-\) 中。为了使目标值变化得更加平滑，我们使用软更新（soft update）：
   \[
   w^- \leftarrow \tau w + (1 - \tau) w^-
   \]
   其中，\(\tau \ll 1\)。这种方法确保了Target Q-Network的权重更新得非常缓慢，从而使目标值 \(y\) 的变化更加平滑和稳定。

5. **稳定性提升：**
   通过引入Target Q-Network，我们将目标值的变化速度控制在一个较低的水平，避免了目标值频繁变化带来的不稳定性。这种设计大大提高了DQN算法的稳定性和收敛速度。

总结起来，创建两个神经网络（Q-Network和Target Q-Network）的主要目的是为了稳定训练过程，避免目标值频繁变化导致的不稳定和震荡，从而提高算法的收敛性和效果。

我：那么，在传统的supervised learning中，例如用linear regression 来预测房价的经典案例中，为什么目标y值就不会变化？是因为它是给定的training set里面的，对吗？

而在这个DQN中，y值不是给定的固定值，而是由网络基于权重来generate的 （类似于是网络自己学习自身，和自己对弈的状态），所以必须要通过增加一个Q网络，避免单一的Q网络自己出现“左右互搏”也就是导致无法学习、无法进步的情况。我的理解正确吗？

GPT：你的理解基本是正确的。让我们更详细地比较一下传统的监督学习和DQN中的目标值问题。

### 传统的监督学习

在传统的监督学习（如线性回归）中：

- **目标值（y）是固定的：** 训练数据集中的目标值 \( y \) 是预先给定的，不会在训练过程中改变。例如，在房价预测的案例中，目标值就是已知的房价数据。
- **固定目标值带来的稳定性：** 由于目标值是固定的，模型只需要不断调整其参数以最小化损失函数（如均方误差）。这种固定的目标值使得训练过程相对稳定，模型可以逐渐收敛到一个较优的解。

### 深度Q学习（DQN）

在DQN中情况有所不同：

- **目标值（y）是动态的：** 目标值 \( y \) 是基于当前Q网络的输出计算的，而Q网络的权重在每次迭代中都会更新。这意味着目标值本身是动态变化的。
- **动态目标值带来的不稳定性：** 由于目标值在每次迭代中都可能变化，这会导致训练过程中的不稳定和震荡。模型在调整参数时，目标值也在变化，可能导致模型难以找到一个稳定的优化方向。

### 引入Target Q-Network的原因

为了应对上述问题，引入了Target Q-Network：

- **稳定目标值：** Target Q-Network的权重更新得更慢，使得目标值 \( y \) 的变化更加平稳，从而避免了频繁变化的目标值带来的不稳定性。
- **避免“左右互搏”：** 就像你提到的，这种设计避免了单一Q网络在训练过程中自我对抗的情况。通过使用一个更新较慢的Target Q-Network，可以使目标值的变化速度减缓，从而使训练过程更加稳定。

### 总结

- **监督学习：** 目标值 \( y \) 是预先给定的固定值，训练过程稳定。
- **DQN：** 目标值 \( y \) 是动态变化的，需要引入Target Q-Network来稳定目标值，避免训练过程中的不稳定性。

你的理解是正确的，通过增加一个Target Q-Network，DQN可以避免单一Q网络在训练过程中自我对抗和不稳定的问题，从而使学习过程更加稳定和有效。

你的问题是非常关键且常见的一个困惑。让我解释一下其中的细节，以澄清你的疑问。

### 初始化时的权重

在开始训练之前，确实会初始化两个网络（Q-Network 和 Target Q-Network），并且它们的权重是相同的。但是，在训练过程中，它们的权重更新方式和频率是不同的，这导致了它们的输出会有所不同。

### 权重更新机制

1. **Q-Network的权重更新：**
   - Q-Network的权重 \( w \) 在每一个训练步骤中都会通过梯度下降算法进行更新。这意味着在每一个mini-batch训练之后，Q-Network的权重都会发生变化。

2. **Target Q-Network的权重更新：**
   - Target Q-Network的权重 \( w^- \) 并不是在每一个训练步骤中更新的。通常是每隔固定的时间步（例如，每1000步）才会将Q-Network的权重复制到Target Q-Network中。
   - 这种较慢的更新频率使得Target Q-Network的权重相对稳定，从而提供了一个稳定的目标值 \( y_i \)。

### 训练过程中的差异

由于Q-Network的权重 \( w \) 是在每一个训练步骤中更新的，而Target Q-Network的权重 \( w^- \) 是在固定的时间步之后才更新，这就导致了在大多数时间里，这两个网络的权重是不同的。

### 计算目标值和损失

在一个mini-batch中：

1. **计算目标值 \( y_i \)：**
   - 使用Target Q-Network计算目标值 \( y_i \)：
     \[
     y_i = r_i + \gamma \max_{a'} \hat{Q}(s'_i, a'; w^-)
     \]

2. **计算当前Q值 \( Q(s_i, a_i; w) \)：**
   - 使用当前的Q-Network计算每个经验的Q值 \( Q(s_i, a_i; w) \)。

3. **计算损失函数：**
   - 使用均方误差（MSE）计算损失：
     \[
     \text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - Q(s_i, a_i; w) \right)^2
     \]

### 为什么MSE不为0

由于Q-Network的权重在每一步都在更新，而Target Q-Network的权重只有在固定的时间步之后才更新，这就导致了它们的输出不会完全相同。因此，目标值 \( y_i \) 和当前Q值 \( Q(s_i, a_i; w) \) 之间会有差异，导致损失函数（MSE）不为0。

### 总结

- **初始化时权重相同：** 在训练开始时，Q-Network和Target Q-Network的权重是相同的。
- **训练过程中权重不同：** Q-Network的权重在每一步都更新，而Target Q-Network的权重更新频率较低。
- **输出值不同：** 由于权重更新频率不同，Q-Network和Target Q-Network的输出值不同，导致目标值 \( y_i \) 和当前Q值 \( Q(s_i, a_i; w) \) 之间存在差异。
- **MSE不为0：** 由于上述差异，损失函数（MSE）不会为0。

希望这能澄清你的疑问。如果还有其他问题，请随时提问！

### 但是

那么，为什么在这个案例中，明明两个网络的权重是同步更新的，却也能进行有效的MSE计算呢？（也就是说，MSE不为零）

GPT: 给出了这个

```python
# 计算目标值
max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
y_targets = rewards + gamma * max_qsa * (1-done_vals)

# 计算当前Q值
q_values = q_network(states)
q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                            tf.cast(actions, tf.int32)], axis=1))

# 计算MSE
loss = MSE(y_targets, q_values)

```

我：哦！我懂了。目标值y_i是通过 rewards + gamma * max_qsa * (1-done_vals) 这个式子算出来的，而Q值是Q-Network直接算出来的！这是它俩有差别的原因
