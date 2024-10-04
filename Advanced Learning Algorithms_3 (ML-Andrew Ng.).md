> This is the third part of Siyuan's note on Andrew Ng's "Advanced Learning Algorithms" course.

Let's get started.

# 14 Machine learning development process

> In the next few videos, I'd like to share with you what is like to go through the process of developing a machine learning system. 
>
> So that when you are doing so yourself, hopefully, you'd be in a position to make great decisions at many stages of the machine learning development process.

## 14.1 Iterative loop of ML development

This is what developing a machine learning model often feel like:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240219170545.png"/>

1. Choosing your machine learning model as well as deciding what data to use, maybe picking the hyperparameters, and so on.
2. Implement and train a model, for the first time, it will almost never works as well as you want it to.
3. Implement some dignostics, such as bias and variance.. >>> based on the insights from this step, **go back to first step and do it again**.

Repeat the loop, until you get what you want.

Look at an example to build an **email spam classifier**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240219171012.png"/>

How do you build a classifier to recognize spam (left) versus non-spam (right) emails? >>> **text classification**

- Construct features: take the top 10k words in English, to define $x_1, x_2,...,x_{10000}$  <<< **feature vector**

  - The mechanism: 

    For every given e-mail text (traning example), if a word **appears**, then its corresponded feature's value is **1**; If it **not appear**, then **0 **.

- Or construct features this way: count the number of times a given word appears in the email. If word "buy" appears twice, maybe you want to set this to 2. But not necessarily, because **0** or **1** works decently well.

- Then: train a logistic regression model or a neural network to predict $y$ given these features $x$ 

How to improve this algorithm's performance (if its error was not acceptable initially):

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240219174409.png"/>

- get more spam email (more training data)
- get more features (email routing, refers to the sequence of compute service. Sometimes around the world that the email has gone through all this way to reach you and emails actually have what's called email header information. >>> Sometimes **the path that an email has traveled can help tell you if it was sent by a spammer or not**)
- detect deliberate misspellings (more features).

How can you decide which of these ideas are more promising to work on (right path can accelarate the project 10x faster)?

- Like, for a high bias problem, don't spend a month on collecting more data.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240219174622.png"/>

Doing the iterative loop of machinery and development, you may have many ideas for how to modify the model or the data, and it will be coming up with different diagnostics that could give you a lot of guidance on what choices for the model or data, or other parts of the architecture could be most promising to try.

Next:

> Start describing to you the error analysis process, which has a second key set of ideas for gaining insight about what architecture choices might be fruitful. 

## 14.2 Error analysis

Error analysis may be the **second most useful** diagnostic method.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240219192257.png"/>

- (If your algorithm misclassifies 100 of these 500 cross validation examples) The error analysis process just refers to **manually looking through these 100 examples and trying to gain insights into where the algorithm is going wrong**.

  1. find a set of misclassified examples

  2. Group them into common teams (For example, if you notice that quite a lot of the misclassified spam emails are **pharmaceutical sales** >>> by manually count each of these kind emails.)

  3. figure out what's the **biggest problem(s)** in these misclassified examples.

     here, "pharma" and "steal passwords" are the most biggest problems (they caused most times of misclassification)

- Note: these categories can be **overlapping** (not mutually exclusive). 

- After this analysis, if you find that a lot of errors are **pharmaceutical spam emails**:

  You may decide to:

  - collect more data of pharmaceutical spam
  - some new features related to "name of drugs" or something >>> to help your learning better recognizing this kind of spam

- if related to phishing urls >>> get more data of phishing emails specifically, or create extra feature of this.

both the bias variance diagnostic as well as carrying out this form of error analysis to be really helpful to screening or to **deciding which changes to the model are more promising to try on next**.

Error analysis can be a bit harder for tasks that even humans aren't good at (by contrast, in the tasks that humans are good at, it can be extremely helpful in deciding).

Next:

> When you train a learning algorithm, sometimes you decide there's high variance and you want to get more data for it. Some techniques they can make how you add data much more efficient.

## 14.3 Adding data

> in the next few videos will seem a little bit like a grab bag of different techniques, bacause machine learning is applied to **so many different problems**, where for some humans are great at creating labels, for some you can get more data and for some you can't.

In this video:

> I'd like to share with you some tips for adding data or collecting more data or sometimes even creating more data for your machine learning application. 

When training machine learning algorithms, it feels like always we wish we had even more data almost all the time. So, how to add data:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240220114929.png"/>

* If adding data for all types is too expensive, then an alternative way of adding data might be **to focus on adding more data of the types where analysis has indicated it might help**.

  e.g. pharma spam

  - if you have a lot of unlabeled email data, you can quickly skim through the unlabeled data and **find more examples specifically a pharma related spam**.
  - This could boost the learning algorithm performance much.

* **Data augmentation**: widely used especially for images and audio data.

  **Modify an existing training example to create a new** **training example**. (for example, modify the handwritten letter)

  - Another explaination: modify the input feature $x$, in order to come up with another example that has the **same label**.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240220114852.png"/>

  > by doing this, you're telling the algorithm that the letter A rotated a bit or enlarged a bit or shrunk a little bit it is still the letter A. 
  >
  > And creating additional examples like this helps the learning algorithm, do a better job learning how to recognize the letter A.

  For a more advanced example of data augmentation:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240220115151.png"/>

  take the letter A and place a grid on top of it. 

  - By introducing random warping of this grid, you can take the letter A and introduce warpings of the leather A to create a much richer library of examples of the letter A.
  - Increases robustness of learning algorithm in recognizing the letter A

* Data augmentation for speech:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240220115650.png"/>

  Adding the noisy background, or other disturbing factors for the example.

  > the times I worked on speech recognition systems, this was actually a really critical technique for increasing artificially the size of the training data I had to build a more accurate speech recognizer.

One tip for data augmentation is that the changes or the distortions you make to the data, **should be representative of the types of noise in the test set**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240220120512.png"/>

- If your added examples are irrelevant to the scenario that could happen in test set, then this Data Augmentation is going to be less helpful.

**Data synthesis**: you make up brand new examples from scratch.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240220121059.png"/>

- Example: photo OCR. **How can you train an OCR algorithm to read text from an image like this**?

  What we can do to create new examples:

  1. take different fonts in the computer, type random text in the text editor
  2. Screenshot it, using different colors and different contrasts and very different fonts
  3. You get the synthetic data on the **right** (the left one were real data from real pictures taken out in the world).


  Synthetic data generation has been used most probably for **computer vision tasks** and less for other applications.

"Finding ways to engineer the data used by your system"

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240220121516.png"/>

- Most machine learning researchers attention was on the **conventional model-centric approach** >>> they don't do anything to the data, they pay fully attention to the code.
- Now, sometimes it can be more fruitful to spend more of our time taking a **data-centric approach**.

Next:

> there are also some applications where you just don't have that much data and it's really **hard to get more data**. 
>
> It turns out that there's a technique called **transfer learning** which could apply in that setting to give your learning algorithm performance a huge boost.

## 14.4 Transfer learning: using data from a different task

For an application where **you don't have that much data**, transfer learning is a wonderful technique that lets you use data from a different task to help on your application. 

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240221105702.png"/>

- Background:  you find a very large datasets of **one million images** of cats, dogs, cars, people, and so on, **a thousand classes**. You can then start by training a neural network on this large dataset.

  train the algorithm to take as input an image X, and **learn to recognize any of these 1,000 different classes**.

  The network **may have 5 layers**. So you end up learning parameters for the first layer of the neural network W^1, b^1, for the second layer W^2, b^2, and so on, W^3, b^3, W^4, b^4, and finally W^5, b^5 for the output layer.

- Then, to apply the transfer learning, (for example, if we need to **recognize handwritten digit**) we can do: 

  make a **copy** of this neural network where you would keep the parameters $W^{[1]}, \vec b^{[1]}; W^{[2]}, \vec b^{[2]}; W^{[3]}, \vec b^{[3]}; W^{[4]}, \vec b^{[4]}$, but **for the last layer, eliminate the output layer and replace it with a much smaller output layer with just 10** (corresponded to the 10 digits).

  then, train the neural network to get the new parameters $W^{[5]}, \vec b^{[5]}$ from scratch.

  - **Option 1**: only train the output layers parameters, don't change any parameter of the prvious layer. use an algorithm like Stochastic gradient descent or the Adam optimization algorithm to only update $W^{[5]}, \vec b^{[5]}$

    > Useful for small training set

  - **Option 2**: train all the parameters in the network, but the first four layers parameters would be initialized using the values that you had trained on top.

    > more suiteble for larger training set.

**Intuition of "transfer learning"**:

- the intuition is by learning to recognize cats, dogs, cows, people, and so on, it will hopefully, **have learned some plausible sets of parameters for the earlier layers** for processing image inputs. 
- Then by **transferring these parameters to the new neural network**, the new neural network starts off with the parameters in a much better place so that we have just a little bit of further learning. Hopefully, it can end up at a pretty good model.

The two steps' names:

- First step "**supervised pre-training**": train the neural network on a very large but general dataset.

  > this could really help your neural network's performance a lot!

- Second step "**fine tuning**": take the parameters initialized from supervised pre-training >>> run gradient descent further to fine tune the weights to suit the specific application

One nice thing about transfer learning as well is, **maybe you don't need to be the one to carry out supervised pre-training**, you can just download the neural network that someone else may have spent weeks training and then replace the output layer with your own output layer and carry out either **Option 1** or **Option 2** to **fine tune** a neural network.

But, why does transfer learning even work?

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240221110353.png"/>

- For example, a neural network for cv project may learn to:
  - **Layer1**: recognize edges (basic structure, low-level features)
  - **Layer2**: recognize corners
  - **Layer3**: recognize basic shapes
- by training a neural network to detect things as diverse as cats, dogs, cars and people, you're helping it to learn to detect these pretty generic features of images, and finding edges, corners, curves, basic shapes, that are **useful for other computer vision tasks**.

**One restriction**: the input type x has to **be the same** for the pre-training and fine-tuning steps.

> You can't pre-train a NN to learn images but find-tune it to recognize audios.

To summarize:



- If you get a well-trained neural network that experienced a large dataset (e.g. 1 million images), then fine tune step may only require **50 images** and get a pretty good result.

> One of the things I like about transfer learning is just that one of the ways that the machine learning community has shared ideas, and code, and even parameters, with each other.
>
> In machine learning, all of us end up often building on the work of each other and that open sharing of ideas, of codes, of trained parameters is one of the ways that the machine learning community, all of us collectively manage to do much better work than any single person by themselves can.
>
> I hope that you joining the machine learning community, and someday maybe find a way to contribute back to this community as well.

Next:

> I'd like to share with you some thoughts on the full cycle of a machine learning project. 

## 14.5 Full cycle of a machine learning project

Training a model is just part of the puzzle. Now, let's see what the full cycle of building a ML Project.

Use **speech recognition** as an example to illustrate the full cycle of the machine learning project:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240222115249.png"/>

- **Step 1. Scope the project**: decide what is the project and what you want to work on.

- **Step 2. Collect data**: decide what data you need to train your machine learning system and go and do the work to get the audio and get the transcripts of the labels for your dataset.

- **Step 3. Train model**: train a speech recognition system and carry out error analysis and iteratively improve your model 

  > (e.g. if your model does poorly on examples with car noise, then do data augmentation espicially on adding car noise of the background to the training examples).

- **Step 4. Deploy in production**: the model is good enough to then deploy in a production environment. Afterwards, you **continue to monitor the performance of the system and to maintain the system** in case the performance gets worse to bring us performance back up instead of just hosting your machine learning model on a server.

More about what "deploy" means:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240222120658.png"/>

- Take your ML model and implement it in a server (an inference server, to call your model to make predictions).

  - if an mobile app is created, then once a ML job is created, it then creates API call to your server, to make predictions on server, and return the result back to mobile user interface.

    > API call gives input x, and get returned $\hat y$

- **Software engineering** is needed to manage scaling to a large number of users, to store the data, and to monitor the system to allow us discovering problems in time and to retrain the model timely.

  > if you are deploying your system to millions of people, you may want to make sure you have a **highly optimized implementations** so that the compute cost of serving millions of people is not too expensive.

-  a growing field in machine learning called **MLOps**, Machine Learning Operations >>> refers to the practice of how to systematically build and deploy and maintain machine learning systems. 

  To do all of these things to make sure that your machine learning model is:

  - reliable
  - scales well, 
  - has good laws,
  - is monitored, and then you have the opportunity to make updates to the model as appropriate to keep it running well.

Next:

> there's one more set of ideas that I want to share with you that relates to **the ethics of building machine learning systems**. This is a crucial topic for many applications.

## 14.6 Fairness, bias, and ethics

I hope that if you're building a machine learning system that affects people that you give some thought to making sure that your system is reasonably fair, reasonably free from bias.

Unfortunately in the history of machine learning that happened a few systems, some widely publicized, that **turned out to exhibit a completely unacceptable level of bias**, for example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240223082529.png"/>

- For example, I have a daughter and if she searches online for certain professions and doesn't see anyone that looks like her, I would hate for that to discourage her from taking on certain professions.

There also be some negative use cases for ml apgirithms:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240223082948.png"/>

- deepfake of someone's face in a video.
- generating harmful content.

> But I have killed the project just on ethical grounds because I think that even though the financial case will sound, I felt that **it makes the world worse off** and I just don't ever want to be involved in a project like that.

There is no concrete guidence of "how to keep ethical" (after I read multiple books about philosophy and ethics), so **there will be some general guidence / suggestions**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240223083629.png"/>

> Read these guidelines.

- A diversed team (anging from gender to ethnicity to culture, to many other traits). 

  > having more diverse teams actually causes a team **collectively to be better** at coming up with ideas about things that might go wrong and it **increases the odds that will recognize the problem and fix it** before rolling out the system and having that cause harm to some particular group.

- A literature search on specific area / industry.

- Audit (pre-realease?) the system.

- Mitigation plan: Keep monitoring.

  > if the car was ever in an accident, there was already a mitigation plan that they could execute immediately rather than have a car got into an accident and then only scramble after the fact to figure out what to do.

The issues of ethics, fairness and bias issues we should take seriously. It's not something to brush off. It's not something to take lightly.

Next:

> I have just two more optional videos this week for you on addressing skewed data sets and that means data sets where the ratio of positive To negative examples is **very far from (50, 50)**.

# 15 Skewed datasets (optional)

## 15.1 Error metrics for skewed datasets

If you're working on a machine learning application where the ratio of positive to negative examples is very skewed, very far from 50-50, then it turns out that **the usual error metrics like accuracy don’t work that well**.

Example:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/02/25/17088420645101.jpg)

- You're training a binary classifier to detect a rare disease in patients. For this algorithm, $y$ is equal to 1 if the disease is present and $y$ is equal to 0 otherwise.
  
- If you got 1% error on the test set (99% correct), then for a rare disease, this is **not a such good performance**. 

   Because of this only 0.5% of your patients have this disease, then even if you wrote a program predict $y=0$ all the time, then **this dumb program surpasses your algorithm** because:
   
    - its accuracy: **99.5%**

    - its error: 0.5%

    > This means, you can’t tell if getting one percent error is actually a good result or a bad result.
   
- Another problem: for three different algorithms, which error was 0.5%, 1% and 1.2% respectively, **you can’t tell which one is the best** (because the 0.5% one may only be the “`print(“y=0”)`” one, and possibly, the one with 1% error may be more useful).

    **how to solve this problem**?
    

When working on problems with skewed data sets, we usually use a different **error metric** rather than just classification error to figure out how well your learning algorithm is doing. A common pair of error metrics are **precision** and **recall**:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/02/25/17088485234669.jpg)

- **The confusion matrix**: 2-by-2 table look like this.
- To evaluate your algorithm’s performance on the **cross validation set**, we count up how many examples:
  
    - Actual class $y = 1$, and predicted class $\hat y=1$, maybe **15** examples.

        > True positive

    - $y=1$, and $\hat y=0$, maybe **5** examples.
      
        > False positive
        
    - $y=0$, and $\hat y=1$, maybe **10** examples.
      
        > False negative
        
    - $y=0$, and $\hat y=0$, maybe **70** examples. 
      
        > True negative

 - In this example, the “skew” isn’t as extreme as the previous rare disease example, because here, positive-negative ratio is **25-75**.

 - Defined TP, FP, TN, FN, then we can define the **precision** and **recall**

     - **Precision**: $\frac {TruePositive}{Predicted Positive} = \frac {TP} {TP+FP} = \frac {15}{15+5} = 0.75$

     - **Recall**: $\frac {TruePositive}{ActualPositive} = \frac {TP}{TP+FN} = \frac {15}{15+10} = 0.6$
     
        
     
        
     
        
     
        
     
        
     
        
    
    These will help you detect if the learning algorithm is “just printing $y=0$ all the time”. In this case, **Precision** will be undefined and **Recall** will be $0$.

The 2 concepts, **precision and recall**, makes it easier to spot if an algorithm is both reasonably accurate:

1. when it says a patient has a disease, there's a good chance the patient has a disease, such as 0.75 chance in this example (**precision**).
2. also making sure that of all the patients that have the disease, it's helping to diagnose a reasonable fraction of them, such as here it's finding 60 percent of them (**recall**).

Next:

> let's take a look at how to trade-off between precision and recall to try to optimize the performance of your learning algorithm.

## 15.2 Trading off precision and recall

In the ideal case, we like for learning algorithms that have high precision and high recall. 

- **High precision**: it has accurate diagnoses (when the algorithm say it’s the disease, then it is highly trustable).
- **High recall**:  if there's a patient with that rare disease, probably the algorithm will correctly identify that they do have that disease.

But it turns out that in practice there's often a trade-off between precision and recall:

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/02/25/17088501798207.jpg)

- Now, we want to predict $y=1$ only is we’re very confident. Hence we’re **pulling up the threshold from 0.5 to 0.7**, even to 0.9

    - In this case, the **precision will increase**, because whenever you predict “1”, you’re more likely to be right.

    - But also, **recall is lower**. Because now (for total number of these disease) we’re likely to diagnose fewer of them.

- On the flip side, suppose we want to avoid missing too many cases of the rare disease.

    > this might be the case where if treatment is not too invasive or painful or expensive but leaving a disease untreated has much worse consequences for the patient.
    
    

    - lower the threshold to 0.3. 

    - Now we’ll have **lower precision** and **higher recall**.

- So, for most cases, there is a trade off between precision and recall. We can draw a graph to visualize the relations between the “threshold” and precision & recall:    

    ![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/02/25/17088500876434.jpg)

    The threshold is up to **you**. 

> For many applications, **manually** picking the threshold to trade-off precision and recall will be what you end up doing

If you want to automatically trade-off precision and recall rather than have to do so yourself, there is another metric called the **F1 score** that is sometimes used to automatically combine precision recall to help you pick the best value or the best trade-off between the two:

- If you trained three algorithms like this, it’s hard still, to choose which algorithm to use.
  
    ![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/02/25/17088504446570.jpg)
    
    it may be useful to find a way to **combine precision and recall** into a single score, and pick highest.
    
- **Method 1 (==DON’T use==)**: take the average. NOT good. Because it may recommend you picking the “`print(“y=1”)`” algorithm.    
- **Method 2**: F1 score >>> **the harmonic mean of Precision and Recall**. $F_1 = \frac {1}{\frac 1 2 (\frac {1}{Precision}+ \frac {1}{Recall})} = 2 \frac {Precision \times Recall}{Precision + Recall}$

    this gives a much greater emphasis to if either **P** or **R** turns out to be very small >>> This could tell us which is better.

Next:

> Next week, we'll come back to talk about another very powerful machine learning algorithm. 
> 
> In fact, of the advanced techniques that why we use in many commercial production settings, I think at the top of the list would be **neural networks** and **decision trees**. Next week we'll talk about decision trees, which I think will be another very powerful technique that you're going to use to build many successful applications as well.

# 16 Decision Trees

## 16.1 Decision Tree Model

One of the learning algorithms that is very powerful, widely used in many applications, also used by many to win machine learning competitions is decision trees and tree ensembles.

Example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240304102622.png"/>

> You are running a cat adoption center, and given a few features, you want to **train a classifier to quickly tell you if an animal is a cat or not**.

- Given training examples: **5 cats** and 5 dogs.

- **3 features** $x_1, x_2, x_3$ to help determine whether a cat or not.

  > For now, each of the features $x_1$, $x_2$, and $x_3$ take on only **two possible values**. 
  >
  > We'll talk about features that can take on more than two possible values, as well as continuous-valued features later in this week

**What is a decision tree**? Here 👇 is a model you might get when you train a decision tree on this dataset.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240304103840.png"/>

- Every one of these ovals or rectangles is called a **node** in the tree.
- The procedure of making decision:
  - start from the root node in the tree (**ear shape**), then choose either go right (**floopy**) or left (**pointy**).
  - Then continue to choose go which direction, and go down. Until it gets final decision.
- A little terminology:
  - The node on the top: **Root node**
  - All the oval shaped nodes: **Decision nodes**
  - The nodes at the bottom: **Leaf nodes**

Another possible shapes of decision tree in this example, they may do better or worse:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240304104042.png"/>

The job of the decision tree learning algorithm is:

- out of all possible decision trees, to **try to pick one** that hopefully does well on the training set, and then also ideally generalizes well to new data such as your cross-validation and test sets as well.

Next:

> How do you get an algorithm to learn a specific decision tree based on a training set? Let's take a look at that in the next video.

## 16.2 Learning process

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240304112317.png"/>

- First step: we have to decide what feature to use at the **root node**.

  > Here we decide, the "ear shape" feature

  Then, split them according to the ear shape.

- Second step: focus on the **left part**, to decide what nodes to put over there.

  > Here, we use "face shape" node

  then we pick four training examples with round face shape down to the right, rest of them to the left. >>> According to the labels, left leaf node is "cat" and right one is "not cat". Done with this part.

- Third step: repeat second step, focus on the **right branch** of this decision tree.

- **Step 4**: Each of these nodes is completely pure, meaning that, there's no longer a mix of cats and dogs. We can create these leaf nodes (cat & not cat). 

Through this process, there were a couple of key decisions that we had to make at various steps during the algorithm:



1. **How to choose what feature to split on at each node**? (to maximize the purity >>> next video)

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240304113255.png"/>

   If we have a "cat DNA" feature, this will be a great feature to use as it simplifies the decision tree and maximize the purity.

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240304112826.png"/>

   But now we only have ear shape, face shape, and whiskers. The decision tree learning algorithm has to choose between them - **which of these features results in the greatest purity**?

   > The next video on **entropy**, we'll talk about how to estimate impurity and how to minimize impurity.

2. **When do you stop splitting**? (The criteria we use just now is: **When a node is 100% one class.**)

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240304114432.png"/>

   Alternatively:

   > When splitting and a further results in **the tree exceeding the maximum depth** (maximun depth, a **parameter** that you could decide.)

   - Why we tend to limit the size of a tree?
     1. to make sure for us to tree doesn't get too **big and unwieldy** 
     2. Second, by keeping the tree small, it makes it less prone to **overfitting**.

   Other stop reasons:

   > when improvements in purity score are below a threshold (when splitting more doesn't make obvious progress anymore). 
   >
   > when number of examples in a node is below a threshold (too little).

Andrew:

> If it feels like a somewhat complicated, messy algorithm to you, **it does to me too**.
>
> these different pieces, they do **fit together into a very effective learning algorithm** and what you learn in this course is the key, most important ideas for how to make it work well.

Next:

> the next key decision that I want to dive more deeply into is how do you decide how to split a node. 
>
> In the next video, let's take a look at this definition of **entropy**, which would be a way for us to measure purity, or more precisely, impurity in a node

# 17 Decision tree learning

## 17.1 Measuring purity

> If your examples are neither all dog nor all cats, then how do you **quantify** how pure is the set of examples?

- **Entropy**. The measure of impurity of a set of data.

Example: a set of 6 examples, 3 dogs and 3 cats.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240312172701.png"/>

- $p_1$ is the fraction of examples that are cats, $\frac 1 2$

- The entropy function is conventionally denoted as $H (p_1)$ like the curve. 

  > Here, where given $p_1$ is $\frac 1 2$, the value of $H(p_1)$ is 1. So it's most impure, the entropy is one.

Some different examples:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240312173125.png"/>

- **5 cats and 1 dog**: here, $p_1$ is about 0.83, and $H(p_1)$ is about 0.65
- **6 cats and no dog**: the $p_1$ is 1, then $H(p_1)$ is 0 >>> zero impurity.
- The closer to a 50/50 mix, the higher the entropy is.

Now let's look at the actual function of $H(p_1)$:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240312173816.png"/>

- Define the $p_0$ to be the fraction of examples that are **not cats**.

  $p_0 = 1-p_1$

- **Here is the function**: $H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2 (p_0)$

  We take $\log_2$ just to make the peak equal to **1** .

- **Note**: by convention for the purposes of computing entropy, we'll take "$0\log(0)$" to be equal to 0.

- **It's similar to the definition of logictic loss function**!

  > there is actually a mathematical rationale for why these two formulas look so similar >>> "**Entropy function**"
  >
  > We won't cover it in this class.

Next:

> In the next video, let's take a look at how you can actually use it to make decisions as to what feature to split on in the nodes of a decision tree.

## 17.2 Choosing a split: Information Gain

> When building a decision tree, the way we'll decide what feature to split on at a node will be based on what choice of feature **reduces entropy the most**.

- My understanding: the entropy function $H(p_1)$ , is kind of the **loss function** in decision tree algorithm.

the reduction of entropy is called **information gain**. Let's look at how to compute the information gain and therefore choose what features to use to split on ar each node of the decision tree.

Example:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240313114021.png"/>

- If we had to split using their **ear shape** feature at the root node: 

  - On the left, $p_1$ is  $\frac 4 5$
  - On the right, $p_1$ is  $\frac 1 5$

- If we split using the **face shape** feature:

  - Left: $p_1 = \frac 4 7$
  - right: $p_1 = \frac 3 7$ 

  > the entropy **H** Much higher than using the **ear shape** feature

- Lastly, we use **whiskers** as feature:

  - Left: $p_1 = \frac 3 4$ 
  - Right: $p_1 = \frac 1 4$ 

- How do we choose one of these features to put on the root node? 

  > rather than looking at these entropy numbers and comparing them, it would be useful to **take a weighted average of them**, combine the entropy of left branch and right branch together. 
  >
  > - Because entropy, as a measure of impurity, is worse if you have a very large and impure dataset compared to just a few examples and a branch of the tree that is very impure.
  
  - Compute the weighted average of each feature caused 2 branches:
    - For ear shape feature, compute $\frac 5 {10} H(0.8) + \frac 5 {10} H(0.2)$
    - the following 2 features are shown as follows.
  
  - Then, use the entropy of the root node ($H(0.5) = 1$) to **minus** the calculated entropy for splited result of each feature. Then compare them, to pick one feature that **contributes the most in reducing the entropy**.
  
    > "**The information gain**" - this is the "reducing of the entropy" value.
  
- Then, we decide the **ear shape** feature at the root node contributes **the most information gain (0.28)**.

Why do we bother to compute reduction in entropy rather than just entropy at the left and right sub-branches?

- It turns out that one of the **stopping criteria** for deciding when to not bother to split any further is if the reduction in entropy is too small.

Let's now write down **the general formula** for how to compute information gain:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240313115003.png"/>

- Here, $p_1^{left} = 4/5$​, stands for the fraction of examples in the left subtree that have a positive label (**cats**).

- $w^{left} = 5/10$​, stands for the fraction of examples of all of the examples of the root node that went to the left sub-branch (5 examples).

- Similarly, $p_1^{right}$, $w^{right}$, and $p_1^{root}$.

- **The formula of Information gain**:
  $$
  H(p_1^{root})-(w^{left}H(p_1^{left})+w^{right}H(p_1^{right}))
  $$

Next:

> Let's put all the things we've talked about together into the overall algorithm for building a decision tree given a training set.

## 17.3 Putting it together

Here is the overall process of building a decision tree:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318112418.png"/>

1. Start with all examples at the root node.

2. Calculate **information gain** for all possible **features**, and pick the one with the highest information gain.

3. **Split dataset** into 2 subsets using selected feature, then create left and right branches of the tree.

4. Keep on **repeating splitting process** until stopping criteria is met.

   **Stopping criteria**:

   > - When a node is 100% one class (**pure**).
   > - Or further splitting will make this tree exceeds the **maximum depth** you had set
   > - or **information gain** from a additional splits is lower than the threshold.
   > - or if the **number of examples in a node** is below a threshold

Let's look at an illustration of how this process will work:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318113957.png"/>

1. We started the process at the root node and by comparing the information gain of all three features, we decide the "ear shape" feature is the best to set at the root node.

   > Pointy ear - Floppy ear

2. Focus on the left sub-branch:

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318112805.png"/>

   1. **Unfinished**: there's still a mix of cats and dogs.

   2. Next step: **pick a feature to split on**. So we go through all these features one at a time, compare the information gain.

      - the information gain for **ear shape** will be **0** (because already split based on the ear shape, all these are pointy ear animals)
      - **Face shape** turns out to have highest information gain (compare with whiskers).

      So, we split this branch according to **face shape** feature, and keep concentrating on the left branch:

      <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318113727.png"/>

      - Now, we have a clean branch that all contained examples are **cats**. **Stopping criteria is met**.

      - So we create a **leaf node** that makes a prediction of cat.

        <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318113932.png"/>

   3. For the right sub-branch, we find that it is all **dogs**. **Stop splitting** and put a leaf node here.

      <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318114147.png"/>

   Until now, we have successfully built the **left sub-tree**.

3. turn our attention to building the **right subtree**.

   <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318114352.png"/>

   1. **Unfinished**. Mixture of dogs and cats.

   2. "**whiskers**" feature gives most information gain. Split these five features according to whether whiskers are present or absent.

      <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318114705.png"/>

   3. Now it's clean. You end up with leaf nodes that predict **cat** and **not cat**.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318114800.png"/>

This is the overall process of building the decision tree.

An interesting fact about "**recursive algorithm**":

- after we decided what to split on at the root node, the way we built the left subtree was by **building a decision tree** on a subset of five examples. The way we built the right subtree was by, again, **building a decision tree** on a subset of five examples

- In computer science, this is an example of a **recursive algorithm**.

  > Recursion in computer science refers to **writing code that calls itself**.

That is to say:

> the way you build a decision tree at the root is by building other smaller decision trees in the left and the right sub-branches.

if you're implementing a decision tree algorithm from scratch, then a **recursive algorithm** turns out to be one of the steps you'd **have to** implement.

About stopping criteria:

- By the way, you may be wondering **how to choose the maximum depth parameter**:

  > There are many different possible choices, but some of the open-source libraries will **have good default choices that you can use**.
  >
  > - One intuition is, the larger the maximum depth, the bigger the decision tree you're willing to build. 
  > - This is a bit like fitting a higher degree polynomial or training a larger neural network (increases risk of **overfitting**).

- decide when to stop splitting is if the **information gained** from an additional split is **less than a certain threshold**

Next:

> in the next few videos, I'd like to go into some further refinements of this algorithm. 
>
> So far we've only used features to take on **two** possible values. But sometimes you have a feature that takes on categorical or **discrete values**, but maybe more than two values. 

## 17.4 Using one-hot encoding of categorical features

> In the example we've seen so far each of the features could take on only one of two possible values. 
>
> The ear shape was either pointy or floppy, the face shape was either round or not round and whiskers were either present or absent. 
>
> But what if you have features that can take on **more than two discrete values**?

Here's a new training set for our pet adoption center application:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240323103010.png"/>

- Notice the **"ear shape"** feature not only being "pointy" or "floopy" here. It can either be “pointy", "floopy" or "**oval**".

  > it can take on three possible values instead of just two possible values.

  In this case, we may end up splitting 3 subsets of data using this feature. 

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240318160128.png"/>

- A different way of addressing features that can take on more than 2 values: **one-hot encoding**.

- We split the **ear shape** feature into **3 different features** (pointy ears, floppy ears and oval ears). And for each training example, use **0** or **1** to label whether the feature is shown:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240323102503.png"/>
  
  > Now, each feature only takes on one of two possible values, and so the decision tree learning algorithm that we've seen previously will apply to this data with no further modifications.

In more detail:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240323102904.png"/>

- if a categorical feature can take on **k** possible values ($k=3$ in our example):
- Then we will **replace it** by creating **k** binary features that can only take on the values 0 or 1.

The idea of using one-hot encoding to encode categorical features also works in **neural networks**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240323103445.png"/>

- Now, we have 5 features, three of them for ear shape, one for face shape and one for whiskers. 
- The list of these 5 features can also be fed to a **neural network** or to **logistic regression** to try to train a cat classifier.

So one-hot encoding is a technique that works not just for decision tree learning, but also lets you encode categorical features using **ones** and **zeros**.

Next:

> But how about features that are numbers that can take on any value, not just a small number of discrete values?
>
> In the next video let's look at how you can get the decision tree to handle continuous value features that can be any number.

## 17.5 Continuous valued features

Let's look at how you can modify decision tree to work with features that aren't just discrete values but **continuous values**.

I have modified the data set to add one more feature which is **the weight of the animal**.

> On average, cats are a little bit lighter than dogs.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240325174015.png"/>

- Now, you have to consist splitting on ear shape, face shape, whisker or **weight**. And, if splitting on the weight feature gives better information gain, then you will split on the weight feature. 

**But how to split on the weight feature?**

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240325181827.png"/>

- "Not a cat" examples are plotted on the horizontal axis. And "Cat" examples are plot high on top.

- We should decide a **threshold**. And how to decide, is based on the **information gain**.

- For example, decide the threshold as the 8 pounds:

  - **left subset**: 2 cats; **right subset**: 3 cats and 5 dogs.
  - So,  $H(0.5)-(\frac 2 {10} H(\frac 2 2) + \frac 8 {10} H(\frac 3 8)) = 0.24$ , the information gain is **0.24**

- If we split on the value of **9** pounds (green line):

  - **Left**: 4 cats; **right**: 1 cat and 5 dogs.
  - Now, the information gain is **0.61**. It is much better.

- If we split the value on **13**, then the information gain is **0.40**.

- **one convention would be**:

  1.  to sort all of the examples according to the weight (as we've plotted)

  2. take all the values that are **mid points** between the sorted list of training example as the values for consideration for this threshold

     > This way, if you have **10 training examples**, you will test **9 different possible values** for this threshold

  3. then try to pick the one that gives you the **highest information gain**. make it as a **decision node**, to split the data.

- For this example, we make "**whether ≤ 9 lbs**" as the decision node, to split the data:

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240325182619.png"/>

**To summarize**:

1. consider different values to split on
2. carry out the usual information gain calculation
3. decide to split on that continuous value feature **with the selected threshold** if it gives the highest possible information gain out of all the possible features to split on.

> that's it for the required videos on the core decision tree algorithm After there's there is an optional video you can watch or not that generalizes the decision tree learning algorithm to **regression trees**.

Next:

> So far, we've only talked about using decision trees to make predictions that are classifications predicting a discrete category, such as cat or not cat. 
>
> But what if you have a regression problem where you want to predict a **number**? In the next video. I'll talk about a generalization of decision trees to handle that.

## 17.6 Regression Trees

In this optional video, we'll generalize decision trees to be regression algorithms so that we can predict a number.

Example: we use *ear shape*, *face shape* and *whiskers*, three features as input **X**, to predict the **weight** of the animal **Y**.


![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/2024/03/30/17117742715208.jpg)

- Note: the "**weight**" is no longer an input feature. Instead, it is the target output **Y**.

Let's look what the regression tree is look like:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240423205320.png"/>

- Note that there's **nothing wrong** with a decision tree that chooses to split on the same feature (such as "**Face shape**" here) in both the left and right sub-branches.

- Finally on leaf nodes, we have four groups of animals. Animals within each group have similar weight.

- **The prediction process**: input an animal, when classification process is over, then output **the average value of the final leaf node**.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240423211612.png"/>

  - e.g. when input an animal has pointy ear and round face, then this regression tree outputs $8.35$, which is the **average** of $7.2$, $8.4$, $7.6$ and $10.2$.
  - Other leaf nodes as well.

How to build a decision tree - the key decision is **how you choose which feature to split on**. Here is how you do it:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240424080820.png"/>

- Rather than reduce entropy in the classification problem, in regression tree, we not focus on how to reduce **the variance of the weight** at each of these 6 subsets of the data.

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240424081959.png"/>

  - **Here's the calculation process**: Firstly get variance of each subset. Then multiply it with its corresponded $w^{left}$ or $w^{right}$ . Finally add them.
  - This is the **weighed average variance**, it plays a similar role to the weighted average entropy we used.

- A good way to choose a split would be to just choose the value of the weighted variance that is **lowest**. Or to say, to choose a split that **reduces the largest proportion of the variance at the root node**, which gives you the largest **information gain** (these two sayings are both correct):

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240424082446.png"/>

  - This is why you choose the "**ear shape**" feature to split on.

How to do next: having chosen the ear shape feature to split on, then we would recursively (keep finding another feature that gives the best **information gain** to split on the data).

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240424082446.png"/>

- And you keep on splitting until you meet the criteria for 

  not splitting any further.

Next:

> So far, we've talked about how to train a single decision tree. It turns out if you train a lot of decision trees, we call this an ensemble of decision trees, you can get a much better result. 

## Lab

the algorithm that computes the entropy based on the $p$ value of a node:

```python
def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1- p)*np.log2(1 - p)
    
print(entropy(0.5))
```

The function that splits the traning examples based on the features:

```python
def split_indices(X, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have 
    that feature = 1 and the right node those that have the feature = 0 
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices
```

- The "index feature" is appointing what feature that was chosen to split on.

The function to compute the **weighted entropy** in the splitted nodes:

```python
def weighted_entropy(X,y,left_indices,right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)
    
    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy
```

- $w^{left}$ and $w^{right}$, the proportion of animals in **each node**.
- $p^{left}$ and $p^{right}$, the proportion of *cats* in **each split**.

- `left_indices` and `right_indices` are lists that contain the indices of samples in left and right branch. 
-  `len(left_indices)` , `len(right_indices)` is the samples contained in each branch.
- the parameter `y` is the `y_train` , the list of ground true labels **Y** of each training example.
- `sum(y[left_indices])` means the **number of cats** that in the left branch.

- **The formula of Information gain**:
  $$
  H(p_1^{root})-(w^{left}H(p_1^{left})+w^{right}H(p_1^{right}))
  $$

Now we've calculated the **weighted entropy** in splitting of the left and right branches:
$$
w^{left}H(p_1^{left})+w^{right}H(p_1^{right})
$$
Next, based on the weighted entropy, we calculate the **information gain**:

```python
def information_gain(X, y, left_indices, right_indices):
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X,y,left_indices,right_indices)
    return h_node - w_entropy
```

- `p_node` is the proportion of cat in all the training samples (the cat is encoded as `1` originally).
- `w_entropy` is the weighted entropy of the split.

Try to split the root node with each feature and compare them (choose which feature gives us the highest information gain):

```python
for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")
    
```

The process is **recursive**, which means we must perform these calculations for each node until we meet a stopping criteria:

- If the tree depth after splitting exceeds a threshold
- If the resulting node has only 1 class (Pure)
- If the information gain of splitting is below a threshold

# 18 Tree ensembles

## 18.1 Using multiple decision trees

> One of the weaknesses of using a single decision tree is that the decision tree can be **highly sensitive to small changes in the data**. One solution to make the algorithm less sensitive or more robust is to build a lot of decision trees. That's called tree ensemble.

Example: how sensitive this algorithm can be.

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240520154541.png"/>

- just by changing a single traning example, the highest information gaining feature from "**Ear shape**" changes to the "**Whiskers**" feature.
- This leads to a totally different sub-trees and totally different tree in the end.
- This makes the algorithm not that robust.

A tree ensemble is like:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240520155411.png"/>

1. Each one of these trees is maybe a plausible way to classify cat versus not cat.
2. what you would do is run all three of these trees on your new example, and get them to **vote** on whether it's the final prediction.

- **Example**: if tree1 and tree3 predicts "cat" and tree2 predicts "not cat", then finally, the tree ensemble predicts "**Cat**" as its final output.

Next:

> how do you come up with all of these different plausible but maybe slightly different decision trees, in order to get them to vote?
>
> In the next video, we'll talk about a technique from statistics called **sampling with replacement**. and this will turn out to be **a key technique** in building the ensemble of trees.

## 18.2 Sampling with replacement

What is "Sampling with relacement":

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240520160144.png"/>

1. We have four tokens of colors. 
2. The sampling with replacement is: every time you pick up one sample from these four tokens, mark its color and put it back, then pick another token up again, repeat.

How to apply it in building the ensemble of trees:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240520160728.png"/>

- Target: we are going to construct multiple random training sets that are all **slightly different from our original training set**.

- Like now we have 10 new training examples (**by sampling with replacement 10 times**):

  <img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240520160748.png"/>

  - some of which are repeats.
  - (so) they don't fully cover the 10 training examples of the original training set.

## 18.3 Random forest algorithm

This is how we create our first random forest algorithm. Firstly, we create a "bagged decision tree":

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240521193526.png"/>

1. Repeat "generate a new data set (use sampling with replacement)" and then "train your algorithm in this dataset" for several times, **until you get a bunch of decision trees**.
   - You do this for a total of **B** times (**B** is the number of trees you build in total)
   - Maybe 100 decision trees (B = 100).
2. When you're trying to make a prediction, you get these trees **all votes on the correct final prediction**.

The note on the **number of trees**:

- The increasing number of trees never hurts performance of the algorithm, but it may slow down the computation.
- "that's why I never use say 1000 trees, that just slows down the computation without meaningfully increasing the performance of the overall algorithm"

This 👆 specific way of creating decision tree is called "*bagged decision tree*". One further modification of this algorithm will make it work even better, and turn it into the "random forest".

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240521195634.png"/>

- One sentence to cover: try to *randomize* the feature choice at each node that can cause the set of trees that you learn *more different from each other*.

  > This reduces the repitition of trees, and causes the algorithm to **explore a lot of small changes** to the data already, making it more robust.

- In other words:

  1. pick a random subset of **K** less than N features as allowed features
  2. in these **K** features, choose the one with *the highest information gain* as the choice of feature to use the split

- A typical choice for the value of K would be to choose it to be square root of N $k = \sqrt n$.

- This is the random forest.

> **where does a machine learning engineer go camping**? 
>
> In a random forest. hhhhhhh

Next:

> Beyond the random forest It turns out there's one other algorithm that works even better, which is a **boosted decision tree**.

## 18.4 XGBoost

Today, by far **the most commonly used way** or implementation of decision tree ensembles or decision trees there's an algorithm called XGBoost.

- It runs quickly, its open source implementations are easily used, it has also been used very successfully to win many machine learning competitions as well as in many commercial applications.

**How XGBoost works**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240522165941.png"/>

- In tree ensembling process, now we make it more likely that **we'll pick misclassified examples that the previouse tree did poorly on** when building a *new tree*.

  - In training and education, there's an idea called **deliberate practice**.

    > For example, if you're learning to play the piano and you're trying to master a piece on the piano. Rather than practicing the entire five minute piece over and over, which is quite time consuming, if you instead play the piece and then **focus your attention on just the parts of the piece that you aren't yet** **playing that well in**, and practice those smaller parts over and over. 
    >
    > Then that turns out to be *a more efficient way* for you to learn to play the piano well.

  - So this idea of boosting is similar.

- **In detail**, we will go back to *original training set* and check which examples are misclassified by the tree just built. Then pick another ten examples, but give a heigher change of picking from one of the examples that was marked "False (misclassified)".

  - So this focuses the second decision trees attention via a process like deliberate practice on the examples that the algorithm is still not yet doing well.
  - The mathematical details of exactly how much to increase the probability of picking this versus that example are **quite complex**, but this does not affect us implementing the algorithm.

Now we turn to the most widely used boot algorithm, **XGBoost**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240522170444.png"/>

- In machine learning competitions, XGBoost and DeepLearning algorithm won a lot of competitions.
- a little **technical note**: rather than doing sampling with replacement, XGBoost actually assigns different weights to different training examples.

The details of XGBoost are quite complex to implement, which is why many practitioners will use the open source libraries. **This is all you need to do in order to use XGBoost**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240522170703.png"/>

## 18.5 When to use decision trees

**Decision trees & tree ensembles**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240522171644.png"/>

- If your data looks like a *giant spreadsheet*, then decision tree will be worth considering.
  - Decision tree will work well on: either cotegorical or countinuous valued features, and both classification (predict a discrete category) and regression task (predict a number), as long as stored in a spreadsheet.
- The unstructured work is for neural networks. 
- The decision tree and tree ensembles are typically **very fast to train**. 
- Small decision trees may be human interpretable (bigger trees are not that easy to understand). 
- Tree ensembles may be more computationally expensive than single tree, but if you got adequate computational budget, then use *XGBoost* directly because it works well.

**Neural networks**:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240522171953.png"/>

- One benefit of neural networks is they work with **transfer learning**. Carry out pre-training on a much larger dataset is critical to getting competitive performance.

- It might be easier to train multiple neural networks than multiple decision trees.

  - When you string together multiple neural networks you can train them all together using gradient descent. 

  - Whereas for decision trees you can only train one decision tree at a time.

Supervised and Unspervied learning:

> Supervised learnings need labeled datasets with the labels Y on your training set. There's another set of very powerful algorithms called unsupervised learning algorithms where you don't even need labels Y for the algorithm to figure out very interesting patterns and to do things with the data that you have. 
>
> I look forward to seeing you also in the third and final course of this specialization which should be on unsupervised learning.

## Lab

The dataset used: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

The data engineering step that use *one-hot encoding* technique to encode the original data:

```python
cat_variables = ['Sex',
'ChestPainType',
'RestingECG',
'ExerciseAngina',
'ST_Slope'
]

# This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
df = pd.get_dummies(data = df,
                         prefix = cat_variables,
                         columns = cat_variables)
```

- one-hot encoding aims to transform a categorical variable with `n` outputs into `n` binary variables.

split the "heart disease" (ground truth label Y) and other features used to predict the heart disease. 

```python
features = [x for x in df.columns if x not in 'HeartDisease'] ## Removing our target variable  
```

then split the training and testing set:

```python
X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size = 0.8, random_state = RANDOM_STATE)
# We will keep the shuffle = True since our dataset has not any time dependency.
```

Choose the best values for the hyper-parameters of the model. The hyperparameters we will use and investigate here are:

- `min_samples_split`: The minimum number of samples required to split an internal node.
  - Choosing a higher min_samples_split can reduce the number of splits and may help to reduce overfitting.
- `max_depth`: The maximum depth of the tree.
  - Choosing a lower max_depth can reduce the number of splits and may help to reduce overfitting.

```python
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
```

- `max_depth = 4`
- `min_samples_split = 50`

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240522185512.png"/>

When it comes to the random forest:

- One additional hyperparameter for Random Forest is called `n_estimators` which is the number of Decision Trees that make up the Random Forest.

A note on hyperparameter training: we are searching for the best value one hyperparameter while leaving the other hyperparameters at their **default values**.

- Ideally, we would want to check every combination of values for every hyperparameter that we are tuning.
- If we have 3 hyperparameters, and each hyperparameter has 4 values to try out, we should have a total of 4 x 4 x 4 = 64 combinations to try.
- When we only modify one hyperparameter while leaving the rest as their default value, we are trying 4 + 4 + 4 = 12 results.
- To try out all combinations, we can use a sklearn implementation called GridSearchCV. GridSearchCV has a refit parameter that will automatically refit a model on the best combination so we will not need to program it explicitly. For more on GridSearchCV, please refer to its [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

In XGBoost implenmetation:

- Even though we initialized the model to allow up to 500 estimators (500 trees), the algorithm only fit 26 estimators (over 26 rounds of training).

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240522191457.png"/>

use this expression to return the best training iteration:

```python
xgb_model.best_iteration
```

The best round of training was round 16, with a log loss of 4.3948.

- For 10 rounds of training after that (from round 17 to 26), the log loss was higher than this.
- Since we set `early_stopping_rounds` to 10, then by the 10th round where the log loss doesn't improve upon the best one, training stops.
- You can try out different values of `early_stopping_rounds` to verify this. If you set it to 20, for instance, the model stops training at round 36 (16 + 20).

The training result:

<img src="https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20240522191846.png"/>
