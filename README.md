# Churn-Problem
## Background & Business Case
Churn, or customer attrition, is essentially when a customer stops using a product or service.  The rate at which customers churn is a critical metric, as it has a direct impact on revenue.  Our initial research into the topic of customer churn within the banking industry showed that not enough is being done to retain customers despite mitigation efforts being a recognized marketing expense.  In terms of marketing we can look at customers both new and existing, and break out costs associated with marketing to these groups into three different categories; costs to attract new customers, costs to retain existing customers, and costs to cross sell to existing customers.  Our focus here is, of course, on retaining existing customers -  basically using marketing dollars to mitigate churn.
According to a 2018 Qualtrics Banking report, more than half of customers who left their bank did not get pushed to stay.  Furthermore,  nearly 75% did not tell their bank they were considering leaving.  With the right data, we hope to be able to identify customers most likely to churn for more effective, targeted marketing and ultimately a more efficient use of marketing budgets.  In theory, this would reduce the churn rate and have a positive effect on the bottom line via increased revenue and a reduction in expenses.  This theory leads us to our research question of focus:  “Can internal bank data be leveraged to identify customers most likely to Churn, and allow for greater targeted use of marketing resources?”

## Literature
In terms of churn research and analysis, there has been a significant amount of work done on this topic.  This is not surprising as it is not a new problem and it affects a wide range of businesses.  In fact, any type of business offering a product or service, particularly those with lots of replacements or like competitors, will be impacted.  
We found our dataset on a public data platform - Kaggle, where the dataset was first made available in October 2018.  Since that time, there have been nearly 10,000 views of the dataset and 2,000 downloads.  Despite the large interest in this churn dataset there are only 33 unique contributors. Much of the work has been focused around exploratory data analysis, visualizations, and of course some classification modeling.  Our team has focused on differentiating ourselves from those projects by adding recognition of feature importance and really focusing using those to make business recommendations.



## Data Descriptions
The dataset consists of 10,000 records with 14 different columns.  Most of the attributes are numerical (11) and the remaining are categorical (3).  The target variable is “Exited” which is a binary variable - coded as a “1” if the customer churned, and a “0” otherwise.

### Columns
![columnsForChurnProblem](https://user-images.githubusercontent.com/64316317/98597330-de095080-22a6-11eb-881c-6a9c31b0e131.PNG)

## Exploratory Analysis
### Attribute Exploration
In the exploration section, we found that the majority of the data is from customers with one or two products. And customers with one product, like owning a credit card or having a checking account, have more chances to leave.  We also discovered the distribution of customer’s age in this dataset and it seems like the younger generation has a relatively lower chance to leave the bank.

![attribute_exploration](https://user-images.githubusercontent.com/64316317/98598217-1eb59980-22a8-11eb-83e0-a27538188f15.png)

![attribute_exploration_2](https://user-images.githubusercontent.com/64316317/98598430-7227e780-22a8-11eb-95fb-e9e996bcce69.png)


Next, in the left graph, we discovered that the majority of the data is from customers in France. And in the right graph, it shows that the proportion of female customers churning is greater than the proportion of male customers churning.

We also found that the majority of the customers that churned are those with credit cards. But maybe most of the customers have credit cards, so it might just be a coincidence. And in another graph, you can see that inactive members have a greater churn, which is not surprising. But you can also find that the overall proportion of inactive members is quite high, which is not a good sign.

### Correlation Analysis
In the  correlation heatmap, we didn’t find any highly correlated predictors in this dataset, which is good for our analysis. And the highest is between age and exited, which is 0.29.


### Outliers Analysis
In the outlier analysis, the boxplot shows that there are few outliers in credit score and a great number of outliers in Age. And we will mention how we deal with the outliers in the next section.

## Modifications
### Removing outliers
We calculated the Z score of the numerical variables and we chose the threshold of 3 or -3. If the Z-score value was greater than 3 or less than -3, that data point would be identified as outliers. After  detecting the outliers using Z-score, we wanted to remove these outliers and got the cleaned dataset. The dataset was removed around 200 rows.
### Attribute Removal
We removed the useless columns including row number, CustomerID and Surname to create models. Our target variable was Exited and predictors were the remaining 10 variables.
### Transforming and splitting
Because we used scikit-learn to build our models and scikit-learn could not handle categorical variables, we should transform categorical as numerical values. There were two categorical features in our dataset. One was Geography[‘France’, ‘Spain’, ‘Germany’] and another was Gender[‘Male’, ‘Female’]. These two variables were sufficiently converted to integer through LabelEncoder from sklearn library. Then the three geographies were replaced by 0,1,2 and gender were replaced by 0 and 1. After encoding the categorical features, all values were numerical. 

To overcome overfitting, we split our dataset into 30% testing and 70% training. In order to make models behave well, we standardized our individual features and then changed them as standard normal distribution data. Standard Scalar transformed the data by subtracting the mean of each feature and then scaled data by dividing their standard deviation. We used Standard Scalar class in the preprocessing module to transform X_train and X_test.

### Modeling Approach
We have built 9 different models to analyze our result and identify the best performing model. The following are the models that we used to train and test the data:
Perceptron 
Decision Tree 
Random Forest 
Dense Neural Network
K-Nearest Neighbors
Logistic Regression  
Naive Bayes 
Support Vector Machines (SVM)
Extreme Gradient Boost

The Customer Churn rate detection task is an imbalanced classification problem: we have two classes we need to identify customers who will churn and who will not, with one category representing the overwhelming majority of the data points.
The positive class is greatly outnumbered by the negative class; hence accuracy is not a good measure for assessing the model performance. Recall can be thought of as a model’s ability to find all the data points of interest in a dataset.
In this case, true positives are the correctly identified customers who would leave, and false negatives would be individuals that the model labels as will not churn who actually leave.  Out of predicted positives, how many of them are correctly predicted is given by the precision whereas recall gives us the number of correct predictions out of the actual positives.


### Initial Results

From our analysis we identified 3 best performing models based on accuracy, recall area under the ROC curve.
The ROC curve below shows us the best 3 models:
Random Forest
Extreme Gradient Boosting
Decision Tree


It was observed that the top 2 best performing models are ensemble models. Ensemble modeling is a process where multiple diverse models are created to predict an outcome by using many different modeling algorithms. The ensemble model then aggregates the prediction of each base model and results in one final prediction for the unseen data. The motivation for using ensemble models is to reduce the generalization error of the prediction.
Random forest (bagging technique) consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction. So, in our random forest, we end up with trees that are not only trained on different sets of data but also use different features to make decisions.

Extreme Gradient Boosting (adaptive Boosting) combines multiple weak learners into a single strong learner. The weak learners in AdaBoost are decision trees with a single split.  When AdaBoost creates its first decision stump, all observations are weighted equally. To correct the previous error, the observations that were incorrectly classified now carry more weight than the observations that were correctly classified. Just like Adaptive Boosting, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of changing the weights for every incorrect classified observation at every iteration like Adaptive, Gradient Boosting method tries to fit the new predictor to the residual errors made by the previous predictor. GBM uses Gradient Descent to find the shortcomings in the previous learner’s predictions. 
XGBoost stands for eXtreme Gradient Boosting. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. Gradient boosting machines are generally very slow in implementation because of sequential model training. Hence, they are not very scalable. Thus, XGBoost is focused on computational speed and model performance. 
By combining weak learner after weak learner, our final model is able to account for a lot of the error from the original model and reduces this error over time.
A Decision tree is a flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. A tree can be “learned” by splitting the source set into subsets based on an attribute value test. This process is repeated on each derived subset in a recursive manner called recursive partitioning. The recursion is completed when the subset at a node all has the same value of the target variable, or when splitting no longer adds value to the predictions.  The feature importance is clear, and relations can be viewed easily. Decision trees can handle high dimensional data. In general, the decision tree classifier has good accuracy.
The screenshot below shows us the model comparison results for all the models that we performed with respect to accuracy, precision. Recall, F1 Score and Area under the ROC Curve for the test dataset.

### Oversampling
One scenario that we noticed during the initial modelling scenario is that we had a class imbalance problem. 

The initial training sample in the dataset, as you can see above, had about 20% of the customers who had exited as against the 80% of customers who were retained. We realized that the model might learn from this imbalance and predict with greater accuracy the customers that are more likely to retained. Due to this scenario, the very first model that we built had a recall of only 0.34, not all of the 1’s were captured by the model and 314 false negatives. It was predicting that 314 people who were churning as not churning. This seemed like a significantly huge number. Now our goal was to hit a balance with recall and false negatives by doing oversampling and then studying the results.
Oversampling involves randomly selecting examples from the minority class (here exited = 1 or
churned customers) with replacement and adding them to the training sample. As you can see from the
infographic that the data after sampling was balanced. The class counts after imbalance can be seen in the picture below. 

### Model Selection
In order to come to a balance between recall and false negatives by doing oversampling, we then modelled using another ensemble technique called Bagging or Bootstrap aggregation. This is a process of combining different models to achieve better performance by sampling(here with replacement). The Random Forest model is a classic example of Bagging. Based on the initial results we saw that Random forest was the best performing model.(Accuracy - 86.56%  False Negatives - 314  recall - 0.44).
Below is an example of the results that we obtained after performing random forest post over sampling.

Increase in the accuracy by 1.5%. It had a 5% increase in recall while also capturing enough false negatives i.e., 272. There seemed to be a steep increase in the ability of the classifier to find all the to be churned customers. 

### Findings


290 customers who actually churn are predicted as churned. Bank will invest in these customers to get them to retain them. 272 customers who actually churn are predicted as retained. This would be unseen CTC.


## Recommendations
We have developed a strategy targeting customers with a three prong approach:  targeted newsletters, product promotions, and activity incentives.  Our model indicated that age and credit score were among the top three features of significance, and our exploratory analysis showed that the average age of an exited person is ~45 years old with an average credit score of 645.  This first strategy would be to send age based content - perhaps financial news and industry reports as well as retirement tips for those middle aged individuals focusing on retirement.  Credit boosting tips and techniques would catch the interest of those with average credit scores.  The bottom line here is to appeal and provide content to at risk age and credit groups.
Our models also showed quite clearly that those customers with only one product were much more likely to exit a bank.  The second strategy is to frame promotions around adding other products - checking or savings accounts, or a credit card and show them as an exclusive deal.  Perhaps putting an expiration on the offer will create a sense of urgency to opt in and ultimately increase the number of products customers have.  The more products a customer has, the more hooks the bank has in them, and the harder it will be to exit.
The data also revealed that customers who became inactive were much more likely to leave a bank.  We of course want to encourage our customers to remain active by using their accounts and products.  The third strategy is in the form of incentives, such as offering a one-time bonus for creating a new checking account linked to a direct deposit.  The point here is straightforward - keep activity among customers.  We feel this three prong approach would yield a positive return on investment, due to the strong analytics supporting the targeted customers.
