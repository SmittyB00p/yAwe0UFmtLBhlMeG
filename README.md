# Customer Satisfaction

This project looked to predict the satisfaction of customers of a food delivery service. The dataset is customer survey responses to 6 questions that serve as the predictor variables with the answers being numeric values between 1 and 5 where 1 was a minimum satisfaction level and 5 being the maximum satisfaction level. 

Main Goal:
* To reach a 73%+ accuracy level

Secondary Goal:
* Find the most relevant features that encapsulate the majority of the information that is necessary to make informed predictions. 

## Exploratory Data Analysis

### Predictors
The initial phase of the project looked to see that the values of the questions corresponded to the values that were laid out by the company (1-5) as well as check for missing values and ultimately look at the descriptive statistics.

From there it was on to look at both a pairwise plot and correlation plot to check for correlations between features and the target as well as colinearity between variables. Plotting each question in regards to the satisfaction category via boxplots one can see that most of the questions have significant overlap between categories and that most have few answers that equal 1.

<!-- Insert Correlation graph here-->
![Correlation Matrix](correlation_matrix.png)

Seen in the image above, the highest correlation of any predictor was in the low 40's and there are two predictors that are near 0. At first glance it seems like the two that are near 0 will unlikely be useful, but additional exploration should be done.

Since the correlation plot utilizes the Pearson correlation, which works well on continuous variables, we can use an ANOVA F-test to gain additional information on the usefulness of the predictors to our model.

<!--  Insert p-vals from f_classif -->

Comparing the p-values to the correlation heatmap above we can see that *'my order was delivered on time'* and *'I am satisfied with my courier'* are the two predictors that look to have significance in classifying satisfied and unsatisfied customers.

### Target Variable
The next thing was to look at the distribution of the two classes and to also see if there were any differences between their average and median scores.

<!-- Insert Distribution Graph here -->
![Target Dist](target_distribution.png)

* The distribution of the satisfied and non-satisfied customers were fairly evenly distributed (1 ~ 55% and 0 ~ 45%)
* The scores were what I expected in that the satisfied customers had higher averages than the non-satisfied customers
* The one thing that is interesting and concerning, especially for a food delivery service, was that the question *'contents of my order was as I expected'* averaged a score of ~ 2.5 for both classes of customers...something to look at going forward since this is a food delivery service.

Looking at the average difference of scores-per-question between those satisfied and unsatisfied looked to indicate that the two highly correlated predictors were both .5 point off, on average, whereas the other questions were only slightly different (at most .3 points) or no difference at all. Giving an affirmation that the initial hypothesis was true - only a couple predictors might be useful in predicting customer satisfaction!

  ### Caveat
  Since there might be interactions between variables that we are not seeing it will be wise to use non-parametric models, such as tree based or ensemble models or other linear models that can account for non-linearity in the features, but that will be for the model exploration phase, which is next.

## Modeling
Using a random seed that roughly split the train and test vectors into equal distributions on the target variable (as mentioned above) I was able to expediate the process of picking a handful of methods to explore by using LazyPredictClassifier. The methods that yielded the best results were tree based models indicating that there is in fact some non-linearity in the dataset that we are not seeing right-off-the-bat. We will explore those models, but before that, let us turn to more basic linear models to see what they yield.
  * Logistic Regression - Using l1 and l2 normalization we can look at the coefficients of the models and see which features are influencing the model.
  * QuadraticDiscriminantAnalysis - More flexible than Logistic Regression
  * SVC - 
  * Decision Trees, Random Forest and Bagging - Highly interpretable models to explore the non-linearity that is evidently present.
  * SGD and XGBoost - Tree based but ____

The models that performed the best in that iterative process of finding the seed were GaussianNB, LinearSVC, XGBoost, and RandomForest. Those models were used in two ensemble techniques - Voting and Stacking - as well as hyper-parameter tuned for further exploration and evaluation. 
- The GaussianNB and LinearSVC models were the top performers after this process with ~ 80% recall scores for both classes

Hyperopt was used on the Random Forest and XGBoost models with the Random Forest model performing the best of the two with ~ 67% and ~ 80% for the positive and negative classes repectively

## Feature Elimination
The last task was to see what features were necessary for modeling purposes and which ones needed to be eliminated from the dataset as well as thrown out of the questionaire.

Using Recursive Feature Elimination (RFE) and Recursive Feature Elimination Cross Validation (RFECV) I was able to see that 4,5, or 6 questions were the most the model needed. After further exploration, question 6 was by far the least likely feature to make a difference in the model and question 1 was the next, but one could make an argument for keeping the feature in the dataset using the RFECV approach.

## Conclusion
After all was said and done the final analysis looked to be that a GaussianNB model with question 6 eliminated from the dataset looked to provide an adequate enough performance metric of ~80% and ~69% for the positive and negative class respectively and an ROC score of ~.76.

### Setup

- use Python 3.9.13 kernel when using notebook
- to activate the virtual environment run the command:
`source .venv/bin/activate`
- and then to install the dependencies for the notebook, run the command:
`pip install -r requirements.txt`
- for reproducability use a seed of 4213 in the models.ipynb notebook
