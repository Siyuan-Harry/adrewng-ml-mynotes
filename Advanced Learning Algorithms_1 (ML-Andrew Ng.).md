# 1 Welcome

What will be learned:

1. you learn about neural networks, also called deep learning algorithms, as well as decision trees.
2. practical advice on how to build machine learning systems

With some of the tips that you learn in this course, I hope that you be one of the ones to not waste those six ones, but instead be able to make more systematic and better decisions about how to build practical working machine learning applications.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231018173533.png"/>

- You learn how neural networks work and how to do inference in Week 1.
- next week, you learn how to train your own neural network (if you have a trading set of labeled examples X and y, how do you train the parameters of a neural network for yourself?)
- In the third week, we'll then go into practical advice for building machine learning systems .
- Next: Decision Trees

## 1.2 Neurons and the brain

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231018175010.png"/>

Even though today neural networks, sometimes also called artificial neural networks, have become very different than how many of us might think about how the brain actually works and learns, some of the **biological motivation** still remains in the way we think about artificial neural networks or computer neural networks today.

How the brain works and how that relates to neural networks:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231018175646.png"/>

- I think the first application area that modern neural networks or deep learning had a huge impact on was currently speech recognition
- Then: people still speak of the image net moments in 2012..
- Then: NLP..

So how does the **brain** work?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231018175928.png"/>

- Here's a diagram illustrating what neurons in a brain looks like.
- The chain conduction of nerve signals is the way how human thoughts were made.

A simplified diagram of a biological neuron, compared to a mathematical model of neuron:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231018180719.png"/>

- blue circle denotes the "neuron", and takes numbers as inputs.
- It does some computation and it outputs some other number (0.7 here) which then could be an input to a second neuron shown on the right.
- What these neurons do collectively is: **Input** a few numbers, carry out some **computation** and **output** some other numbers.

Caveat⚠️:  Today we have **almost no idea** how the human brain works, thus attempts to blindly mimic what we know of the human brain today, which is frankly very little, probably **won't get us that far toward building real intelligence**, certainly not with our current level of knowledge and neuroscience.

In fact, those of us that do research in deep learning have shifted away from looking to biological motivation that much but **instead I just using engineering principles to figure out how to build algorithm more effective.**

Next question: **why now?** Why is it that only the last handful of years that neural networks have really taken off?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231018181907.png"/>

- In many application areas, the amount of digital data has exploded

- What we saw was with traditional machine learning algorithms such as logistic regression and linear regression, **even as you fed those albums more data**, it was very difficult to get the performance to keep on going up.

  > "They weren't able to take effective advantage of all this data we had for different applications."

- ......And if you were to train a **very large neural network** meaning one with a lot of these artificial neurons then for some applications the performance would just keep on going up.

  > "if you're able to train a very large neural network to take advantage of that huge amount of data you have, then you could attain performance on **ANYTHING**. -- that just were not possible with earlier generations of learning algorithms."

- And GPU, that was also **a major force** in allowing deep learning algorithms to become what it is today. 

## 1.3 Demand Prediction

> To illustrate how neural networks work, let's start to an example.

Example: Demand Prediction

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231018184412.png"/>

>  In this example, you're selling t shirts and you would like to know if a particular T-shirt will be a top seller, yes or no. And you have collected data of different t shirts that were sold at different prices as well as which ones became a top seller.

- Input feature $x$: price. 
- And if we applying logistic regression, then it fit a sigmoid function to the data and look like this.

But how can we apply neural network to do this prediction?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019095258.png"/>

1. Change the function $f(x)$ to $a$ in alphabet. This term $a$ stands for **activation**. 

   > This is actually a term from neuroscience and refers to how much a neuron is sending a high output to other neurons downstream from it.

   this logistic regression units can be thought of as a **single neuron** in the brain. 

   what the neuron does is:

   > it takes input the price $x$ and then it computes this formula on top and it outputs the number $a$ (the probability of this T-shirt being a top seller).

2. Given this description of a single neuron, building a neural network now just requires taking **a bunch of these neurons** and **wiring them together**.

A more complex example of demand prediction (we're going to have **4 features** to predict whether or not a T-shirt is a top seller):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019100805.png"/>

- Maybe whether or not a T-shirt becomes a top seller actually depends on few factors: 
  - First, what is the affordability of this T-shirt? 
  - Second, what's the degree of awareness of this T-shirt that potential buyers have?
  - Third is, perceive quality.
- What we're going to do is create **one artificial neuron** to try to estimate the probability that this T-shirt is perceived as **highly affordable** and affordability is mainly a function of price and shipping cost.
  - Second, I'm going to create another artificial neuron here to estimate is there **awareness** in this case is mainly a function of the marketing of the T-shirt.
  - Finally we're going to create another neuron to estimate do people perceive this to be of **high quality** and that may mainly be a function of the price of the T-shirt and of the material quality.
- Given these estimates of affordability, awareness and perceive quality, we then wire the outputs of these three neurons to another neuron on the right.

In the terminology of neural networks, we're going to group these three neurons together into what's called a **layer** :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019101043.png"/>

- A layer is a grouping of neurons which take us **input** the same or similar features and that in turn **output** a few numbers together.
- A layer can also have single neuron. This layer on the right is also called the **output layer** .

"Activations": In the terminology of neural networks, we're also going to call affordability, awareness and perceive quality to be **activations**.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019101609.png"/>

- So these numbers on affordability, awareness and perceived quality are the **activations** of these three neurons in this layer (blue ones).
- And: also this output probability is the activation of this neuron shown on the right.

So this particular neural network could therefore carries out computations as follows:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019101547.png"/>

- It inputs four numbers.
- then this layer of the neural network uses those phone numbers to compute three new numbers also call activation values.
- And then the final layer, the output layer of the neural network uses those three numbers to compute one number. 
- And in a new network this list of four numbers is also called the **input layer** and that's just a list of four numbers.

How does this truly works: one simplification.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019101823.png"/>

- The way a neural network is implemented in practice, each neuron in a certain layer will have access to every feature to every value from the previous layer (here, **every neuron** in middle layer has the access to **every value** from the input layer).
- You can imagine that, if you're trying to predict **affordability** and it knows what's the price, shipping cost, marketing and material maybe or learn to ignore marketing material and just figure out through setting the parameters appropriately to **only focus on the subset of features that are most relevant to affordability**.

To further simplify: **vectorization**.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019102617.png"/>

- These 4 input features can be write down as a vector $\vec x$.
- Then, computes three activation values, which is, $\vec a$.
- Then, finally outputs the probability of T-shirts being a top seller, $a$.

The layer in the middle, is called, "**hidden layer**" (corresponded to "input layer" and "output layer").

- That's because, correct values for "affordability", "awareness" and "percieved quality" are **hidden**, so you don't see them in training set.

Another way of thinking about neural network:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019103622.png"/>

- Rather than using the original features, price, shipping, cost, marketing and so on, using a new (maybe better set of) features: **affordability, awareness and pursue quality** that are hopefully **more predictive** of whether or not this T-shirt will be a top seller.

- So, one way to think of this new network is, just logistic regression but it is a version of logistic regression that **can learn its own features** that makes it easier to make accurate predictions. (Actually, we **don't** assign features like **affordability, awareness and pursue quality** for this algorithm!)

  > What in neural network does is instead of you needing to manually engineer the features, it can learn its own features to make the learning problem easier for itself. "hidden"

Let's take a look at some other examples of neural networks, specifically examples with more than one hidden layer:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019104118.png"/>

- when you're building your own neural network two of the decisions you need to make are:

  1. How many hidden layers do you want?
  2. How many neurons do you want each hidden layer to have?

- This question of how many hidden layers and how many neurons per hidden layer is a question of the **architecture of the neural network** you learn (later in this course).
- in some of the literature, you see this type of neural network with multiple layers like this called a **multi layer perceptron**. So if you see that, that just refers to a neural network that looks like what you're seeing here on the slide 👆.

Next video: computer face recognition.

## 1.4 Example: Recognizing Images

If you're building a face recognition application, you might want to train a neural network that takes this **input** the picture like this and **outputs** the identity of the person in the picture:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019105053.png"/>

This image is 1000 by 1000 pixels and so its representation in the computer is actually as 1000 by 1000 grid (**matrix** of pixel intensity values) .

- In this example, my pixel intensity values or pixel brightness values goes from 0 to 255
- 197 here would be the brightness of the pixel and the very upper left of the image..and so on...Down to 214, would be the lower right corner of this image.
- So, you can transfer this image as a **vector** with 1,000,000 values.

Then, train a neural network that takes as **input** a feature vector with 1 million pixel brightness values and **outputs** the identity of the person in the picture:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019105305.png"/>

One interesting thing, when you trying to visualize a neuron network has trained with a lot of pictures, then:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019110056.png"/>

- In the earliest layers of a neural network, you might find that the neurons are looking for very **short lines or very short edges** in the image.
- If you look at the next hidden layer, you find that these neurons might learn to group together lots of little short lines, a little short edge segments in order to look for **parts of faces**. (Upper left, look like to detect the presence of an **eye**, in the certain position..)
- And in the next layer, trying to detect larger **face shapes**..

So these little neurons visualizations actually correspond to **differently sized regions in the image**.

let's see what happens if you were to train this new network on a different dataset:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231019110244.png"/>

Later this week you see how you can build a neural network yourself and apply it to a **handwritten digit recognition application**.

In the next video, let's look more deeply into the concrete mathematics and the concrete implementation of details of how you actually build one or more layers of the neural network.

# 2 Neural network model

## 2.1 Neural network layer

> The fundamental building block of most modern neural networks is a **layer of neurons**.
>
>  In this video, you learn how to **construct a layer of neurons** and once you have that done, you'll be able to take those building blocks and put them together to form a large neural network.

This is what we saw in last lesson (for demand prediction example):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231022181544.png"/>

**Zoom in to the hidden layer to look at its computations:**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231022181745.png"/>

- This hidden layer inputs 4 numbers
- these 4 numbers are inputs to each of three neurons
- Each of these three neurons is just implementing a little logistic regression units (function).

Each neuron has its own parameters $w$ and $b$ , and it outputs the activation value $a$ (here perhaps $a_1=0.3$):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231022182133.png"/>

Other 2 neurons do same thing, and then they output a vector $\vec a$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231022182439.png"/>

Then this $\vec a$ is passed to the final output layer of this neural network.

Here comes some notation:

- $\vec a^{[1]}$ refers to the $\vec a$ output by the **layer 1** (**the output of layer 1**).

- Then: add superstrip square brackets to denote activation values of the hidden units of a specific layer (here, **layer1**).

- the thing to remember is: whenever you see this **[1]** that just refers to a quantity that is associated with **layer 1**.

  > **Similarly for other layers as well..**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231022183249.png"/>

This output of layer 1 $\vec a^{[1]}$ becomes **the input to layer 2**.

Then, let's zoom into the computation of **layer 2** of this neural network:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231022195715.png"/>

What it computes: $a_1^{[2]}=g(\vec{w_1}\cdot \vec{a}^{[1]} + b_1)$ 

- and to notice: this new $a_1$ is a scalar value, not a vector.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231022200725.png"/>

Once the neural network has computed $a^{[2]}$ there's one final optional step:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231022201106.png"/>

- if a **binary prediction 1 or 0** is expected, then using 0.5 as a threshold, this model makes prediction.
- Otherwise, if you want the probability, then $a_1^{[2]}$ is the answer

**Recap**: In neural network, every layer inputs a vector of numbers and applies a bunch of logistic regression units to it and then computes another factor of numbers that then gets passed from layer to layer until you get the final output layer's computation, which is **the prediction of the neural network**.

**Next**: let's go on to use this foundation we build now to look at some even more complex/larger new network models.

## 2.2 More complex neural networks

This is what we gonna use in this lesson:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231029213244.png"/>

Zoom in to layer3:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231029220316.png"/>

* Layer3 **inputs** a vector $\vec a^{[2]}$ that was computed by the previous layer, and **outputs** $\vec a^{[3]}$  which is another vector.

* three neurons has parameters w 1 b 1, w 2, b 2 and w 3 b 3. And they compute three a values $a_1$ $a_2$ $a_3$, to construct $\vec a^{[3]}$ .

* Notice:  $\vec w_1^{[3]}$ meaning the parameters associated with layer3, dot product $\vec a^{[2]}$ , which was the output of layer 2 which became the input to layer 3. 

  > This is why $[3]$ here: they are parameters asocciated with layer3. 

  So, the neurons update the value of $\vec a$ , for example, from $a^{[layer2]}$  to $a^{[layer3]}$.

* Blue circled: hide the superscripts and subscripts associated with the second neuron, fill by yourself.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231031192037.png"/>

More general form of this equation: **Layer $l$ and unit $j$** .

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231031194737.png"/>

- $g$ here is the sigmoid function. In the context of a neural network, $g$ has another name which is also called the activation function (because it outputs the activation value).

  > Not only sigmoid function can be the activation function, other function can be plugged in as well.

- Last to emphasize: input $\vec x$ is also called $a^{[0]}$ . So this same equation also works for the first layer where when $l = 1$. 

  > So, the activations of the first layer that is $a^{[1]}$ would be the sigmoid times the weights dot product with $a^{[0]}$, which is just this input feature vector X.

  "sigmoid times" ? 🤔️

- So with this notation you now know how to compute the activation values of any layer in the neural network as a function of the parameters as well as the activations of the previous layer.

Next:

> Let's put this into an inference algorithm for a neural network. In other words, how to get a neural network to make predictions. Let's go see that in the next video.

## 2.3 Inference: making predictions (forward propagation)

> Let's take what we've Learned and put it together into an algorithm to let your neural network **make inferences** or make predictions. This would be an algorithm called **forward propagation**.

So it's just a binary classification problem where we're gonna input an image and classify is this the digit 0 or the digit 1. 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231031195432.png"/>

For the example on this slide, I'm going to use an 8 by 8 image (upper right).

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231031201015.png"/>

- so this image of "**1**" is this grid or matrix of 8 by 8, 64 pixel intensity values.
- 255 denotes a **bright** white pixel and 0 would denote a **black** pixel.
- Different numbers are different shades of gray in between the shades of black and white.
- Given these 64 input features, we're going to use a neural network (left half of the slide) with **2 hidden layers** where the first hidden layer has **25 neurons**, second hidden layer has **15 neurons**. And then finally the output layer outputs what's the chance of this being 1 versus 0.

 So let's step through the **sequence of computations** of the neural network:

1. Go from $\vec x$ to $a^{[1]}$ , which is the first hidden layer does: 

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231031201326.png"/>

   > "I've written $\vec x$  here but I could also written $a^{[0]}$ here. They are in the same meaning."

2. The next step is to compute $\vec a^{[2]}$ 

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231031201716.png"/>

3. Finally, compute $a^{[3]}$, which is not a vector ( just a scalar number).

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231031201742.png"/>

   $a^{[3]}$ is also **the output of the neural network**. So you can also write that as $f(x)$ .

   > Remember when we Learned about linear regression and logistic regression, we use F of X to denote the output of linear regression or logistic regression. So we can also use F of X to denote **the function computed by the neural network** as a function of  X.

Because this computation goes from left to right, you start from X then compute $\vec a^{[1]}$, then $\vec a^{[2]}$, then $a^{[3]}$. This algorithm is also called **forward propagation** because you're propagating the activations of the neuron.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231031202110.png"/>

- So you are making these computations in the forward direction from left to right.
- This is in contrast to a different algorithm called **backward** propagation or back propagation (will learn next week).

Some tips in choosing neural network architecture:

- This type of neural network architecture where you have more hidden units initially and then the number of hidden units decreases as you get closer to the output layer. 
- This is a **pretty typical choice** when choosing your network architectures

Next lesson:

>  Now that you've seen the map and the algorithm, let's take a look at how you can actually implement this in **Tensorflow**. Specifically, let's take a look at this in the next video.

# 3 TensorFlow implementation

## 3.1 inference in Code

> Tensorflow is one of the leading frameworks for implementing deep learning algorithms. When I'm building projects Tensorflow is actually the tool that I use the **most often**.
>
> The other popular tool is **pytorch** but we're going to focus in this specialization on **tensorflow**. In this video, let's take a look at how you can implement inferencing code using Tensorflow. 

One of the remarkable things about neural networks is the same algorithm can be applied to so many different different applications, So, here is another application case:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102185522.png"/>

Can a learning algorithm help **optimize the quality of the beans you get** from a roasting process like this?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102185625.png"/>

When you're roasting coffee, two parameters you get to control are the **temperature** at which are heating up the raw coffee beans to turn them into nicely roasted coffee beans, as well as the **duration** or how long are you going to roast the beans.

So:

1. Temperature
2. Duration time

And in this chart:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102185522.png"/>

* **Blue circle** means the good taste coffee.
* **Red Cross** means the bad taste coffee.

So: 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102190259.png"/>

- if you cook it at **too low temperature**, it doesn't get roasted and it ends up undercooked.
- And if you cook it **not for long enough,** the duration is too short, it's also not a nicely roasted set of beans.
- And finally, if you were to cook it **either** for **too long** or for **too high temperature**, then you end up with overcooked beans (little bit burnt beans), so there's not good coffee either.

So the task is: given a feature vector X with both temperature and duration, how can we do inference in a neural network to get it to tell us **whether or not this temperature and duration setting will result in good coffee or not**.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102200045.png"/>

So, we're going to set:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102200817.png"/>

1. ==[Line 1]== X to be an array of two numbers. The input features 200 degrees Celsius and 17 minutes.

2. ==[Line 2]== And layer 1 (units = 3 and activation = 'sigmoid'), "**Dense**" here is just the name of this layer.

   > "**Dense** is another name for the layers of a neural network that we've Learned about so far. 
   >
   > As you learn more about neural networks, you learn about other types of layers as well"

3. ==[Line 3]== Then finally, to compute the activation value $a^{[1]}$

Then: compute the $a^{[2]}$ by input $a^{[1]}$ to the layer 2. Finally if you wish the threshold is at 0.5 then you can just test the output number of this network:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102201055.png"/>

So that's how you do inference in the neural network using Tensorflow.

Some additional details in the **lab**:

- How to load the Tensorflow library
- How to also load the parameters $w$ and $b$ 

So please be sure to take a look at the lab.

Let's look at one more example and we're gonna go back to the handwritten digit classification problem:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102202000.png"/>

In this example:

- $\vec x$ is a list of the pixel intensity values
- Layer 1 is a dense layer with 25 units
- Then, compute $a^{[1]}$ by applying layer 1 to the input $\vec x$

Similarly, build layer 2, layer 3...then finally you can optionally threshold $a^{[3]}$ to come up with a binary prediction for $\hat y$.

So that's the syntax for carrying out inference in Tensorflow. 

> Tensorflow treats data in a certain way that is important to get right. In the next video, let's take a look at how Tensorflow handles data.

learning notes from lab:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102210501.png"/>

- **Tensorflow and Keras**
  Tensorflow is a machine learning package developed by Google. In 2019, Google integrated Keras into Tensorflow and released Tensorflow 2.0. Keras is a framework developed independently by François Chollet that creates a simple, layer-centric interface to Tensorflow. This course will be using the **Keras interface.**

-  A **tensor** is another name for an **array**.

- What are weights and bias? Weights is $w$, bias is $b$.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102211500.png"/>

- Procedure of using tensorflow (in sigle liner regression computing):

  1. define a layer
     ```python
     linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
     ```

  2. Set the instantiation of the weights $w$ and $b$

     ```python
     a1 = linear_layer(X_train[0].reshape(1,1))
     
     #or:
     set_w = np.array([[200]])
     set_b = np.array([100])
     linear_layer.set_weights([set_w, set_b])
     ```

  3. compute a value $x$ from the training set.
     ```python
     a1 = linear_layer(X_train[0].reshape(1,1))
     ```

     or, compute the whole traning set straightforwardly:

     ```python
     prediction_tf = linear_layer(X_train)
     ```

- Logistic regression in tensorflow:
  $$ f_{\mathbf{w},b}(x^{(i)}) = g(\mathbf{w}x^{(i)} + b) \tag{2}$$
  where:
   $$g(x) = sigmoid(x)$$ 

  1. Set the logistic neuron.
     ```python
     model = Sequential(
         [
             tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
         ]
     )
     ```

     Tensorflow is most often used to create multi-layer models. The **[Sequential](https://keras.io/guides/sequential_model/) model** is a convenient means of constructing these models.

  2. `model.summary()` shows the layers and number of parameters in the model.
     
     <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231102212741.png"/>
     
     There is only one layer in this model and that layer has only one unit. Then you can get layers and weights:
     ```python
     logistic_layer = model.get_layer('L1')
     w,b = logistic_layer.get_weights()
     ```
     
  3. set weight and bias
     ```python
     set_w = np.array([[2]])
     set_b = np.array([-4.5])
     # set_weights takes a list of numpy arrays
     logistic_layer.set_weights([set_w, set_b])
     ```
  
  4. Run the layer and check the output
     ```python
     a1 = model.predict(X_train[0].reshape(1,1))
     print(a1)
     ```

Congratulations! You built a very simple neural network and have explored the similarities of a neuron to the linear and logistic regression from Course 1.

## 3.2 Data in Tensorflow

> This video:  step through with you **how data is represented in numpy and in tensorflow.** So as you're implementing new neural networks, you can have a consistent framework to think about how to represent your data.

Unfortunately: there are some **inconsistencies** between how data is represented in numpy and in tensorflow. 

- So it's good to be aware of these conventions, so that you can implement correct code and hopefully get things running in your neural networks.

How tensorflow represents data:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106192213.png"/>

- Why this double square bracket in `np.array` ?

Some knowledge about metrics:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106194020.png"/>

- 2*3 matrix >>> In code, to store this matrix:

  ```python
  np.array([[1,2,3],
            [4,5,6]])
  ```

  > So a matrix is just a 2D array of numbers.

  This is where these double square bracket comes from.

- 4*2 matrix >>> A matrix can also be other dimensions like 1 by 2 or 2 by 1...

More examples:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106194839.png"/>

- setting `x = np.array([[200,17]]) ` >>> 1 row, 2 columns

  > "row vector"

- setting `x = np.array([200],[17])` >>> 2 rows, 1 column

  > "column vector"

- But for `x = np.array([200,17])`, with only one square bracket, this only create **1D array**, **a vector** 

  > not one by two of two by one, it's just a **linear array**. 
  >
  > A **list of numbers** with **no rows or columns**.

But in **Tensorflow**: the convention is to use **matrices** to represent the data. Because the Tensorflow was designed to handle very large datasets. And by representing the data in matrices instead of 1D arrays, it lets Tensorflow be a bit more computationally efficient internally. 

Going back to the first example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106195228.png"/>

Understanding how this neural network works with data in code:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106195247.png"/>

- If you print out $\vec a^{[1]}$ , you'll discover this is a **1*3 matrix** `[[0.2,0.7,0.3]]` (the result of layer1's computing). 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106200627.png"/>

- And "**Tensor**" here is a data type that the Tensor Flow team had created in order to store and carry out computations on matrices efficiently. That's just a way of representing **matrix**.

  > Likewise, the `ndarray` is the way of numpy to represent the matrix.

- How to convert these 2 datatypes:

  - `a1 = layer1(x)` converts a **ndarray** to a **tensor**, by computing one layer using numpy array as input.
  -  `a1.numpy()` converts a **tensor** `a1` to numpy **ndarray**.

Now let's take a look at what the activations output by the **second layer** look like:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106201311.png"/>

- If you print out $\vec a^{[2]}$, you'll discover that's a tensorflow **tensor**, with just one number 0.8..

> I'm used to loading data and manipulating data in **Numpy** but when you pass a numpy array into Tensorflow Tensorflow likes to **convert it to its own internal format**, the tensor and then operate efficiently using **tensors** and when you read the data back out you can keep it as a tensor or convert it back to a **numpy array**.
>
> When you convert back and forth, whether you're using a numpy array or a tensor is just **something to be aware of** when you're writing code.

Next, let's take what we've learned and put it together to actually build a neural network. Let's go see that in the next video.

## 3.3 Building a neural network

> Let's put it all together and talk about how to build a new network in Tensorflow.

What you saw earlier:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106201938.png"/>

It turns out that Tensorflow has **a difference way** of implementing for a prop as well as learning:

Now, instead of you manually taking the data and passing it to layer 1 and then taking the activations from layer 1 and passing to layer two, we can instead tell Tensorflow that we would like it to take layer 1 and layer 2 and **string them together to form a neural network.** 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106202218.png"/>

That's what the **Sequential()** function in tensorflow does..

With the sequential framework, Tensorflow can do a lot of work for you:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106203413.png"/>

1. Create layer1 and layer2 of the model, by calling `Dense()` function.

2. Call `Sequential()` function to string these two layers together to create the neural network modal.

3. Take the training datas inputs $x$ and put them into a numpy array (a **4*2 matrix**).

4. Write the target value $y$ as another numpu array (a **1D arrary** 1,0,0,1 corresponding the four training samples).

5. Call `model.compile()` function (more in next week)

6. Call `model.fit(x,y)` function (more in next week)

   > This tells Tensorflow to take this neural network that is created by **sequentially string together** layers 1 and 2 and to train it on the data X and Y.

7. If you have a new example, say `x_new` (numpy array with these two features), you just have to call `model.predict(x_new)`.

**That's it.** This is pretty much the code you need in order to train as well as do inference on a neural network in Tensorflow.

Let's redo this for the **digit classification example** as well:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106203807.png"/>

Instead of assigning layer 1, layer 2, layer 3 explicit like this, we would more commonly just **take these layers and put them directly into the sequential function**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106203921.png"/>

So, this is the more commonly used form of code (except of model constructing, the rest of the code works same as before):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231106204020.png"/>

> One thing I want you to take away from the machine learning specialization is the ability to use cutting edge libraries like Tensorflow to do your work efficiently. 
>
> But I don't really want you to just call 5 lines of code and not really also know what the code is actually doing underneath the hood.
>
> So in the next video, I'd like to go back and share with you **how you can implement from scratch by yourself for propagation in Python** so that you can understand the whole thing for yourself in practice.
>
> Let's also go through what it would take for you to implement for a propagation from scratch because that way, even when you're calling a library and having it run efficiently and do great things in your application, I want you in the back of your mind to also have that deeper understanding of what your code is actually doing.

# 4 Neural network implementation in Python

> If you had to implement for a propagation yourself **from scratch in Python**, how would you go about doing so?
>
> I don't really recommend doing this for most people, but maybe someday someone would come up with an even better framework than Tensorflow and Pytorch. And whoever does that may end up having to **implement these things from scratch themselves**.

## 4.1 Forward prop in a single layer

There will be quite a bit of code, and these code can be seen later in **optional lab / practice lab**. 

> And the goal of this video is to just show you the code to make sure you can understand what is doing. **Don't worry about taking detail notes on every line.** If you can read through the code on this slide and understand what is doing, that's all you need.

Let's take a look at how you implement for a prop in a single layer (still coffee roasting model):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107114108.png"/>

* using 1D array in python to represent all of these vectors and parameters.

* The first value needed to compute: $\vec a^{[1]}$ , this vector is divided into three numbers: $a_1^{[1]}$ , $a_2^{[1]}$ , $a_3^{[1]}$ .

  * $a_1^{[1]}$ : 

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107105921.png"/>

    Then compute $z_1^{[1]}$ >>> compute $sigmoid(z_1^{[1]})$ :

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107113238.png"/>

  * next, $a_2^{[1]}$ (written in code, `a1_2`) >>> the same way of computing $a_1^{[1]}$, and $a_3^{[1]}$ :

    <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107113718.png"/>

    > Wait, where does the $w$ and $b$ comes from?

* Then, grouping them together in a numpy array to form $\vec a^{[1]}$:
  
  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107113842.png"/>
  
* Next step: compute  $\vec a^{[2]}$.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107113927.png"/>

That's it. That's how you implement Forward Prop using **just Python and Numpy**.

> Let's in the next video look at how you can simplify this to implement forward prop for a more general neural network rather than hard coding it for every single neuron like we just did.

## 4.2 Neural network implementation in Python

> Let's now take a look at the more general implementation of forward prop in Python.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107190818.png"/>

Steps:

1. define a **dense() function**, which takes us input the activation `a_in` from the previous layer as well as the parameters $w$ and $b$ for the neurons in a given layer.

   > what this function will do is **input** a the activation from the previous layer and we'll **output** the activations from the current layer.

   - What is `W`: a 2 by 3 matrix, the 3 columns are $\vec w_1^{[1]}$, $\vec w_2^{[1]}$, and $\vec w_3^{[1]}$ .
     
     <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107191128.png"/>
   - `b` is a 1D array, contains $\vec b_1^{[1]}$, $\vec b_2^{[1]}$, and $\vec b_3^{[1]}$ .
     
     <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107191740.png"/>
   - `a_in` here could be $\vec a^{[0]}$ , also called as input vector $\vec x$.
     
     <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107191726.png"/>

2. Here is the code of **dense function**:
   
   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107191931.png"/>
   
   * `units = W.shape[1]` : `W` is a 2*3 matrix, so `shape[1]` returns **number 3**, its colums, equal to the number of units in this layer..
   * `a_out` initializes the value of output $\vec a^{[1]}$.
   * then a **for loop** calculates three output `a` values one by one: $\vec a_1^{[1]}$, $\vec a_2^{[1]}$, $\vec a_3^{[1]}$, then tie them together to form the **final output value** of this function $\vec a^{[1]}$.
   
3. String three `dense()` function together to form a neural network model (forward prop).
   
   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231107192723.png"/>
   
   This is relatively simpler. Compute $\vec a^{[1]}$ to $\vec a^{[4]}$ respectively, and output $\vec a^{[4]}$.
   
   - A little notation about `W`:
   
     > Notice that here I'm using a **capital W** because under the notational conventions from linear algebra is to use **uppercase** when referring to a **matrix** and **lowercase refer to vectors and scalers**.

**That's it!** You now know how to implement forward prop yourself from scratch and you get to see all this code and run it and practice it yourself in the **practice lab**.

Andrew's reminders:

> Your ability to understand what's actually going on would make you **much more effective** when debugging your code.

To recap & next video:

> That's the last required video of this week with code in it. In the next video, I'd like to dive into what I think is a fun and fascinating topic, which is what is the relationship between neural networks and AI or AGI

# 5 Speculations on artificial general intelligence (AGI)

> Ever since I was a teenager starting to play around with neural networks, I felt the dream of maybe someday building an AI system that's as intelligent as myself or as intelligent as a typical human. That was one of the most inspiring dreams of AI.

## 5.1 Is there a path to AGI?

> But let's take a look at what this AGI, artificial general intelligence dream is like .

AI actually includes two very different things: ANI & AGI

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231108163255.png"/>

- **ANI:** This is an AI system that does one thing, a narrow task, sometimes really well and can be incredibly valuable.
- **AGI:** This hope of building AI systems, they could do anything a typical human can do.

Why neural network model **cannot** actually simulate th human brain:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231108163445.png"/>

Reasons:

1. A logistic regression unit is really **nothing like** what any biological neuron is doing (so much simpler).
2. Even to this day, I think we have almost **no idea** how the brain works.
   
   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231108163643.png"/>

"given our very limited understanding both now and probably for the near future of how the human brain works, I think just trying to simulate the human brain as a path to AGI will be an **incredibly difficult** path."

One thing to keep the hope alive: "Some experiments on animals, shows or suggests that the same **piece of** biological brain tissue can do a surprisingly **wide range of task**." 

- Hypothesis:  may be a lot of intelligence could be due to one or a small handful of learning algorithms >>> all we need to do is to figure it out.

Detail of experiment:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231108164143.png"/>

- if we feed the visual output to the Auditory cortex (which originally designed to "**hear**"), and cut the audio input wire, then the Auditory cortex **learns to see**.

Another example: touch >>> see

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231108164402.png"/>

This suggests:

> Many different parts of the brain just depending on what **data** it is given can learn to see or learn to feel or learn to hear as if there was maybe one algorithm that **just depending on what data it is given learns to process that inputs accordingly**.

Is that "decoder-only"?

See this:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231108164700.png"/>

Human brain is unlimited. Human brain is amazingly adaptable.

>  I still find this one of the most fascinating topics and I still often ideally think about it in my spare time and maybe someday you would be the one to make a contribution to this problem.

Next:

> In particular, in the optional videos to come, I'd like to share you some details of how to implement vectorize implementations of neural networks.

# 6 Vectorization (optional)

## 6.1 How neural networks are implemented efficiently

> One of the reasons that deep learning researchers have been able to scale up neural networks and build really large neural networks over the last decade is because neural networks can be **vectorized**. 
>
> And it turns out that parallel computing hardware, including GPUs, but also some CPU functions are **very good at doing very large matrix modifications**.

We can replace For loops to Vectorized calculation, to **improve the calculating efficiency** (using example in **4.2**):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231108170102.png"/>

- This is a dense layer calculation using **For loops**:
  
  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231108170249.png"/>
  
- This is the same calculation using **Vectorization**:
  
  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231108170321.png"/>
  
  - `np.matmul()` is how **numpy** carries out **matrix multiplication**.
  
  - Now, all of these quantities, `X` which is fed into the value of `A_in` as well as `W`, `B`, as well as `Z` and `A_out`. All of these are now 2D arrays >>> **Matrices**.
  
    > Also the reason why they're capitalized.

So, this is code for vectorized implementation of forward prop in a neural network.

> But what is this code doing and how does it actually work and what is this maturl actually doing? >>> Next video.

## 6.2 Matrix multiplication

> So you know that the matrix is just a block or 2D array of numbers. What does it mean to multiply two matrices?

let's start by looking at how we take dot products between **vectors**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231206165744.png"/>

Here is what happens when taking dot product:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231206165856.png"/>

In a more general case:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231206165949.png"/>

You compute **Z** by:

> multiplying the first element together and then the second elements together and the third and so on, and then adding up all of these products. 

So that's the **Vector Dot product.**

The formula for calculating the dot product is:
$$
\mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} a_i b_i
$$
In which the $\mathbf{A}$ and $\mathbf{B}$ are vectors，and $a_i$, $b_i$ are the elements of vectors.

Another **equivalent way** of writing a dot product:

1. **First**, transpose a vector (turn this into a row)
2. And it turns out that if you **multiply** a transpose with the $\vec w$, it is the same as taking the **dot product** between $\vec a$ and $\vec w$.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231206180636.png"/>

To recap:

> **z** equals the **dot product** between $\vec a$ and  $\vec w$ is the same as **z** equals $\vec a$ **transpose** multiply by $\vec w$. So these are two ways of writing the exact same computation to arrive at **z**.

Now let's look at **vector matrix multiplication** (take a vector and multiply it by a matrix): 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231206180510.png"/>

Here:

- $\vec a$ is a vector (2 by 1 matrix).
- $\vec a^{T}$ is also a vector (1 by 2 matrix).
- $W$ is a 2 by 2 matrix.

If we want to compute $Z = \vec a^{T} W$ , then:

- **Z** turns out to be a **1 by 2 matrix**.

That's how it computes:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231206181524.png"/>

- **First element of Z**: multiplying $\vec a^{T}$ by the first column of $W$.

- **Second element of Z**: multiplying $\vec a^{T}$ by the second column of $W$.
- So, $Z = [11\,\,\,17]$ 

One last thing: to take **vector matrix multiplication** and generalize it to matrix multiplication.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207094957.png"/>

How to multiply $A^T$ and $W$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207092738.png"/>

1. Call the **columns** of  $\vec a_1$ and $\vec a_2$ , which means, the **rows** of transposed $A$ are: $\vec a^T_1$  and $\vec a^T_2$ .
2. Same as before, **columns** of $W$: $\vec W_1$ and $\vec W_2$ .

Then, start to do this matrix multiplication:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207095651.png"/>

> Andrew: I encourage you to think of the columns of matrices **as vectors**.

- First: pay attention to the first row of $A^T$ ($\vec a_1^T$) and multiply that with $W$.

  And this produces **first row** of the target $Z$:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207095258.png"/>

  - Step 1. $\vec a^T_1$ dot product with $\vec W_1$ 

  - Step 2. $\vec a^T_1$ dot product with $\vec W_2$

  > That's exactly what we did on the previous slide.

- Next: take $\vec a_2^T$ and multiply with $W$ ($\vec a_2^T$ times $W$) , to produce second row of target $Z$ :

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207100019.png"/>

  - Step 1. $\vec a^T_2$ dot product with $\vec W_1$ , which is
    $$
    (-1 \times 3) + (-2 \times 4) = -11
    $$

  - Step 2. $\vec a^T_2$ dot product with $\vec W_2$ , which is
    $$
    (-1 \times 5) + (-2\times6) = -17
    $$

- So this produces:
  $$
  Z = \begin{bmatrix}
  11&17 \\
  -11&-17 \\
  \end{bmatrix}
  $$

Let's talk about the general form of a matrix matix multiplication is defined in **next video**!

## 6.3 Matrix multiplication rules

We'll try to multiply matrix $A$ and matrix $W$.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207103014.png"/>

As before, I encourage you to think of the columns of this matrix as three vectors: $\vec a_1$, $\vec a_2$, $\vec a_3$ .

- Then we'll take transpose of this matrix, to get $\vec a_1^T$, $\vec a_2^T$, and $\vec a_3^T$.

Then, see $W$:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207111446.png"/>

> notice that I've also used slightly different shades of orange to denote the different rows of $A^T$, and different shades of blue to denote different columns of $W$ .

So, how to compute $A^T$ times $W$ to get $Z$:

- The **columns** of $W$ will influence the columns of $Z$ correspondingly. (**4** columns)
- **Rows** of $A^T$ will influence the rows of $Z$ correspondingly. (**3** rows)

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207114101.png"/>

So, $Z$ will be a **3 by 4** matrix.

Let's start off and figure out how to compute the number in the first row and the first column of $Z$:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207114300.png"/>

-  $\vec a_1^T \times \vec w_1$  is the first row of $A^T$ times first column of $W$ .
  
  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207114654.png"/>

Then, how to compute number in row 3, column 2 of $Z$ :

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207114749.png"/>

Now, grab row 3 of $A^T$ and column 2 of $W$ and dot product those together.

- Here:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207141453.png"/>

- And the computation:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207141330.png"/>

  (My question here: what if their dimensions dosen't match? >>> **NOT** able to multiply)

One more example: what is row 2 and column 3 of matrix $Z$ ?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207141709.png"/>

- It's $ (-1 \times 7) + (-2 \times 8) = -23$ 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207142031.png"/>

Then, you can compute all the elements of matrix $Z$:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207142123.png"/>

The last interesting fact about vector and matrix dot products: **you can only take dot products between vectors that are the same length.**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207142313.png"/>

Two properties of matrix multiplication:

- Matrix multiplication is valid only if the number of **columns** of the first matrix ($A^T$) is equal to the number of **rows** of the second matrix $W$ . 

  > So that when you take dot products during this process, you're taking dot products of vectors of the **same size**.  

- The output of this multiplication will have the same number of rows as $A^T$ and the same number of columns as $W$.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231207142736.png"/>

Next: let's take what we've learned about matrix multiplication and apply back to the vectorized implementation of a neural network. That'll be **really cool**. It makes neural networks run much faster.

## 6.4 Matrix multiplication code

First, write down matrix $A$, $A^T$ and $W$ using numpy: 

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/01/17041184047725.jpg)

- Here notice: a easier way to create $A^T$ is to use numpy’s function: `AT = A. T` .

And then, calculate them using `np.matmul(AT, W)`, to get the result:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/01/17041184047966.jpg)

- By the way, `Z = AT @ W` can do the same job as `np.matmul(AT, W)`.

Then, let's look at what a **vectorized implementation of forward prop** looks like:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/01/17041225197580.jpg)

- How to get matrix $W$ and $B$: take the parameters $\vec w_1^{[1]}$, $\vec w_2^{[1]}$ and $\vec w_3^{[1]}$, then stack them in columns, to get matrix $W$. Matrix $B$ the same way.

- Now the parameters of this layer are $W$ and $B$ (vectorized!), then, we input $A^T = [200, 17]$ to this layer to get $A^{[1]}$.

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/01/17041225197657.jpg)

The computation flow:

1. Compute $Z$ from given input $A^T$. Here we get `Z = [165, -531, 900]`, this includes:
    - $z^{[1]}_1 = 165$
    - $z^{[1]}_2 = -531$
    - $z^{[1]}_3 = 900$
2. Apply sigmoid function to this $Z$, which is $g(Z)$, ends up being `[1, 0, 1]`.

Now, let's look at how you implement this in code:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/01/17041218663083.jpg)

- The 2 differences are (orange color notes):
  
    ![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/01/17041218663273.jpg)
    
    1. Now we use `AT` instead of `A` (In TensorFlow by convention, we call this just `A`)

    2. Now  in the definition of dense layer, we use `AT`. However, this is called `a_in` in TensorFlow.
    
    > “And there is a convention **in TensorFlow** that:  individual examples are actually **laid out in rows** in the Matrix X rather than in the Matrix X transpose.”
    

This explains why with just a few lines of code, you can implement forward prop in the neural network and moreover get **a huge speed bonus**. 

Because matrix multiplication can be done very efficiently using fast hardware and get a huge bonus, because modern computers are very good at implementing matrix multiplations such as `np.matmul()` efficiently.

> Next week, we'll look at how to actually **train** a neural network.

# Lab

- If network has $s_{in}$ units in a layer and $s_{out}$ units in the next layer, then 
        - $W$ will be of dimension $s_{in} \times s_{out}$.
        - $b$ will a vector with $s_{out}$ elements

- Therefore, the shapes of `W`, and `b`,  are 
    - layer1: The shape of `W1` is (400, 25) and the shape of `b1` is (25,)
    - layer2: The shape of `W2` is (25, 15) and the shape of `b2` is: (15,)
    - layer3: The shape of `W3` is (15, 1) and the shape of `b3` is: (1,)

>**Note:** The bias vector `b` could be represented as a 1-D (n,) or 2-D (1,n) array. Tensorflow utilizes a 1-D representation and this lab will maintain that convention. 

# 7 Neural Network Training

## 7.1 TensorFlow implementation

> Last week you Learned how to carry out inference in the neural network. This week we're going to go over training of a neural network.

Let's continue with our running example of **handwritten digit recognition**, recognizing this image as 0 or 1. Using this neural network architecture:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/02/17041670283000.jpg)

- First hidden layer with 25 units.
- Second hidden layer with 15 units.
- Finally, one output unit.

The code we can use in TensorFlow to train this network:

- This first part may look familiar from the previous week, nothing new.

    ![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/02/17041672507788.jpg)

- **Second step**: ask TensorFlow to compile the model (by specifying the **loss function**).

    > The key step in asking Tensorflow to compile the model is to specify what is the loss function you want to use. Here, we use “Sparse categorical cross entropy”.
    > 
    > See more in next video.
    

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/02/17041680029048.jpg)

- **Third step**: call the fit function, specifying the **training set** and **epochs**.

    > Tell the TensorFlow to fit the model that you specify in Step 1 using the loss of the cost function that you specified in Step 2 to the **dataset** (x, y).
    >
    > What’s more, “**epochs**” is a technical term for how many **steps** of learning algorithm like grading descent you may want to run.
    
    ![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/02/17041679923182.jpg)
    

Next: we will dive more deeply into what these steps in the TensorFlow implementation are actually doing.

> I hope that you be able to not just call these lines of code to train the model, but that you also understand what's actually going on behind these lines of code.

## 7.2 Training Details

Let's take a look at the details of what the TensorFlow code for training a neural network is actually doing!

Firstly: let’s recall how you trained a **logistic regression model** in the previous course.

This procedure can be divided into 3 steps:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/02/17041802573171.jpg)

1. **Step 1**: To specify how to compute the output given the input feature $\vec x$ and parameters $\vec w$ and $b$ (the function).
2. **Step 2**: To specify the loss function and cost function. 
    - **Loss function** was a measure of how well is logistic regression doing on a single training example $(x,y)$.
    - Cost function was to compute an average of every loss of total $m$ training examples (entire training set).
3. **Step 3**: To use an algorithm (gradient descent) to minimize that cost function $J(\vec w, b)$. To minimize it as a function of the parameters $w$ and $b$.
    - $w$ and $b$ is updated simultaneously.

The same three steps is how we can train a **neural network** in TensorFlow.

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/02/17041809181640.jpg)

1. **Step 1**: specifying the function (construct our model).
    ```python
    model = Sequential([
            Dense(…)
            Dense(…)
            Dense(…)
    ])
    ```
2. **Step 2**: Compile the model and specify the loss function (here, “Binary cross entropy” loss function), and the cost function is constructed automatically (just taking average loss from the entire training set).
    ```python
    model.compile(loss = BinaryCrossentropy())
    ```
3. **Step 3**: To call the function to minimize the cost.
    ```python
    model.fit(X, y, epochs = 100)
    ```

Let's look in **greater detail** in these three steps in the context of training a neural network:

First, specify how to compute the output given the input $\vec x$ and parameters $\vec w$ and $b$, to **create the model**.

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/02/17041811731674.jpg)

- The code snippet specifies the entire architecture of the neural network (3 dense layers, number of neurons of each layer, and activation function type).

Then, In the second step, you have to specify what it is the **loss function**, and that will also define the **cost function**.

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/03/17042481821125.jpg)

- For the Mnist digit classification problem is a binary classification problem, and most common loss function to use is this one (binary cross entropy).

    ```python
    model.compile(loss = BinaryCrossentropy())
    ```
    
    > it’s actually the **same** loss function as what we had for **logistic regression**.
    
- and, don’t forget to import this function at first:
  
    ```python
    from tensorflow.keras.losses import BinaryCrossentropy
    ```
    
    > “**Keras**” was originally a library that had developed independently from TensorFlow, but eventually it got merged into TensorFlow.
    
- Having specified the **loss** with respect to a single training example, Tensorflow knows that the **cost** you want to minimize.
- Optimizing this cost function will result in fitting the neural network to your binary classification data.

> By the way, I don't always remember the names of all the loss functions in Tensorflow, but I just do a quick web search myself to find the right name and then I plug that into my code.

Otherwise, if you want to solve a **regression** problem, you then need to minimize the **squared error loss**. Then you can use this in TensorFlow (at the bottom of this PPT): 

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/03/17042531563970.jpg)

- the function is:
  
    ```python
    model.compile(loss = MeanSquredError())
    ```
- And don’t forget to import the function initially:
  
    ```python
    from tensorflow.keras.losses import MeanSquredError
    ```
  This denotes the cost function (upper right):

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/03/17042536805233.jpg)

- Capital $W$ may include $W^{[1]}$, $W^{[2]}$ and $W^{[3]}$, and $B$ as well.
- And the $f_{w,b}(\vec x)$ denotes the output function of neural network as well.

“binary cross entropy loss function”, **where does this name come from**?

- In statistics, this (written on top left) is called “cross entropy loss function”.
- The “binary” here is to emphasize this function is to solve the **binary** classification problem.

Finally, you will ask TensorFlow to minimize the cost function (gradient descent):

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/03/17042545536895.jpg)

If you are using gradient descent to train the parameters of a neural network, then:

- you would repeatedly for every layer $l$ and for every unit $j$, update $w^{[l]}_j$ based on its cost function, and similarly for the parameters $B$ as well. 
- And after doing, say, 100 iterations of gradient ascent, hopefully you get to a good value of the parameters.
- So, the key thing you need to compute is these **partial derivative terms**, and TensorFlow would handle this using the “**back propagation**” algorithm.
- And, all you need to do with the code, is to call `fit()`.
  
    ```python
    model.fit(X, y, epochs = 100)
    ```
    
    > In fact, what you see later is that TensorFlow can use an algorithm that is even a little bit faster than gradient descent, and you see more about that later this week as well.

In fact, most commercial implementations of neural networks today use a library like TensorFlow or PyTorch.

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/01/03/17042546864891.jpg)

> But as I've mentioned, it's still useful to understand how they work under the hood so that if something unexpected happens, which still does with today's libraries, you have a better chance of knowing how to fix it.

“Now that you know how to train a basic neural network, also called a **multilayer perceptron**, there are some things you can change about the neural network that would make it even more powerful… 

In the next video, let's take a look at how you can swap in different activation functions as an **alternative to the sigmoid activation function** we've been using. This will make your neural networks work even much better.”

---

> This is the end of the first part of Siyuan's note on Andrew Ng's "Advanced learning algorithms" course.

Go on to the next part!