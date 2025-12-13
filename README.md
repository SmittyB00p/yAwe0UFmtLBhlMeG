# Customer Satisfaction

This project looked to predict the satisfaction of customers of a food delivery service. The dataset is customer survey responses to 6 questions that serve as the predictor variables with the answers being numeric values between 1 and 5 where 1 is a minimum satisfaction level and 5 the maximum satisfaction level. 

Main Goal:
* To reach a 73%+ accuracy level

Secondary Goal:
* Find the most relevant features that encapsulate the majority of the information that is necessary to make informed predictions. 

## Exploratory Data Analysis

### Predictors
The initial phase of the project looked to see that the values of the questions corresponded to the values that were laid out by the company (1-5) as well as check for missing values and ultimately look at the descriptive statistics.

From there it was on to look at both a pairwise plot and correlation plot to check for correlations between features and the target as well as colinearity between variables. Plotting each question in regards to the satisfaction category via boxplots one can see that most of the questions have significant overlap between categories and that most have few answers that equal 1.

<!-- Insert Correlation graph here-->
![Correlation Matrix](/images/correlation_matrix.png)

Seen in the image above, the highest correlation of any predictor was in the low 40's and there are two predictors that are near 0. At first glance it seems like the two that are near 0 will unlikely be useful, but additional exploration should be done.

Since the correlation plot utilizes the Pearson correlation, which works well on continuous variables, we can use an ANOVA F-test to gain additional information on the usefulness of the predictors to our model.

<!--  Insert p-vals from f_classif -->
| Feature                                | f-statistic         | p-value              |
|----------------------------------------|---------------------|----------------------|
| my order was delivered on time         | 10.561708467842157  | 0.001486386824045104 |
| I am satisfied with my courier         | 6.5826903496397575  | 0.01148805555119031  |
| the app makes ordering easy            | 3.58689768005047    | 0.06056651807995719  |
| I ordered everything I wanted to order | 2.886975954772599   | 0.09180590776658196  |
| I paid good price for my order         | 0.5166974960654367  | 0.47360621658268376  |
| contents of my order was as I expected | 0.07308433330563574 | 0.787347251607012    |

Comparing the p-values to the correlation heatmap above we can see that *'my order was delivered on time'* and *'I am satisfied with my courier'* are the two predictors that look to have significance in classifying satisfied and unsatisfied customers.

### Target Variable
The next thing was to look at the distribution of the two classes and to also see if there were any differences between their average and median scores.

<!-- Insert Distribution Graph here -->
![Target Dist](/images/target_distribution.png)

* The distribution of the satisfied and non-satisfied customers were fairly evenly distributed (1 ~ 55% and 0 ~ 45%)
* The scores were what I expected in that the satisfied customers had higher averages than the non-satisfied customers
* The one thing that is interesting and concerning, especially for a food delivery service, was that the question *'contents of my order was as I expected'* averaged a score of ~ 2.5 for both classes of customers...something to look at going forward since this is a food delivery service.

Looking at the average difference of scores-per-question between those satisfied and unsatisfied looked to indicate that the two highly correlated predictors were both .5 point off, on average, whereas the other questions were only slightly different (at most .3 points) or no difference at all. Giving an affirmation that the initial hypothesis was true - only a couple predictors might be useful in predicting customer satisfaction!

  ### Caveat
  Since there might be interactions between variables that we are not seeing it will be wise to use non-parametric models, such as tree based or ensemble models or other linear models that can account for non-linearity in the features, but that will be for the model exploration phase, which is next.

## Modeling
Using a random seed that roughly split the train and test vectors into equal distributions on the target variable (as mentioned above) I was able to expediate the process of picking a handful of methods to explore by using `LazyPredictClassifier`. 

The methods that yielded the best results were tree based models indicating that there is in fact some non-linearity in the dataset that we are not seeing right-off-the-bat. We will explore those models, but before that, let us turn to more basic models to see what they yield.

  * Logistic Regression
  * QuadraticDiscriminantAnalysis
  * SVC
  * Decision Trees, Random Forest and Bagging
  * SGD and XGBoost

<!-- Insert model scores here -->
<!-- ![Models Scores](/images/) -->



## Feature Elimination

<!-- Insert features graph here -->
<!-- ![Used Features](/images/) -->

<!-- The above bar graph shows the features that were selected by the models chosen for experimentation.

We can see that the most used feature was *'my order was delivered on time'* followed by *'I am satisfied with my courier'* and thirdly, for all intensive purposes *'the app makes ordering easy'*. This validates our initial hypothesis from the exploratory data phase and gives us justification in saying that the question that can be dropped from the next survey is that of *'I paid good price for my order'*. -->

## Conclusion

<!-- Main Goal:
* Achieve 73%+ accuracy score

Achieved:
* From the ______ model we achieved a _______ score which _______ our original goal.

Secondary Goal:
* Find most relevant features

Achieved:
* A confirmation that the two features that were most heavily used in classifying customers were *'my order was delivered on time'* and *'I am satisfied with my courier'*. And a justification in dropping the question *'I paid good price for my order'* from the next survey. -->

### Setup
- create a virtual environment (venv) with any name (customary to use .venv for virtual environment name):
  `python3.9 -m venv .venv`
- to activate the virtual environment run the command:
  `source .venv/bin/activate`
- and then to install the dependencies for the notebook, run the command:
  `pip install -r requirements.txt`
- use `.venv Python 3.9` kernel in notebooks
- for reproducability use a seed of 5249 in all notebooks
