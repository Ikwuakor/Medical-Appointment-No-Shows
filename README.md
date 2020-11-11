# Predicting Medical Appointment No Shows

>The following project explores a dataset from Brazil to determine to what extent we can predict whether patients will show up to their medical appointments as scheduled. There are 110,527 medical appointments with 14 associated variables. The target variable is ``no-show``, which records whether the patients show-up or are no-shows to their appointment.
> ### Scholarship variable explanation:
> Bolsa Família (Family Allowance) is a social welfare program of the Government of Brazil, part of the Fome Zero network of federal assistance programs. Bolsa Família provides financial aid to poor Brazilian families.
***
Patients failing to make their appointments can be quite problematic for medical professionals. From wasted personnel on staff to other patients, who were otherwise available, being scheduled for a later date, missed appointments can be quite disruptive for medical offices. In fact, missed appointments might also be disruptive to the absent patient's own good health! The question is: how do medical professionals combat missed appointments? Is there a way to predict ahead of time who is most likely to miss their appointments? The following analysis could serve as:
+ a method to help Brazilian medical offices anticipate appointment no-shows and subsequently determine whether intervention methods are cost effective.
+ a template for similar investigations across various locations, both inside and outside of Brazil, as well as in other professions where appointments are scheduled (with industry relevant variables).

### Let's check for, and examine, any outliers in our lone continuous variable, $Age$. Let's plot our age variable.

![image](https://user-images.githubusercontent.com/42311832/98757522-f2724980-2389-11eb-9e17-31bc1b57edd3.png)

According to our boxplot, there are some outliers on the higher end of our ``Age`` variable. The distribution of ages seem to be fairly well represented across most twenty year ranges, showing a higher frequency around zero years and falling off after 60 years. Let's take a closer look using z scores.

> + number of Outliers above 1 threshold: 19762
> + number of Outliers above 2 threshold: 1746
> + number of Outliers above 3 threshold: 5
> + number of Outliers above 4 threshold: 0

## Examining our categorical variables

![image](https://user-images.githubusercontent.com/42311832/98757972-e1760800-238a-11eb-894f-4523c1f580ec.png)

We have an unbalanced dataset with most observations from all our categorical variables falling into the 'No' category, including our target variable ``is_noshow``. What's interesting is that almost 80% of all our observations made their appointments. Therefore, going forward, **80% will be the baseline accuracy metric**, ie. the accuracy score we can expect running a naive model. However, since our data is unbalanced, there are other metrics which we'll prioritze when measuring the success of our models. More on that later.
### Now, let's take a look at the disparities between no-shows within each of the categorical variable above.

![image](https://user-images.githubusercontent.com/42311832/98758112-387bdd00-238b-11eb-887b-0788eabf4ed7.png)

Here, we get a better sense of how the categorical variables relate to our target variable. The cancellation ratios are very similar for ``Alcoholism`` and ``is_male`` to the overall no-show percentage of 20%, with the other variables showing slightly larger differences between *show-ups* and no-shows when the condition in question was present (irrespective of the other variables). The largest disparity occurs with ``SMS_received``, at a 10.9% difference, followed by ``Scholarship`` at 3.9%. These variables might lend the most predictive power towards determining whether a no-show will occur. From the ``is_noshow`` grouped means below, we can verify that the greatest difference occurs, among the binary variables, with the ``SMS_received`` variable.

## Examining our Neighbourhood variable

![image](https://user-images.githubusercontent.com/42311832/98758198-70832000-238b-11eb-8b97-665ba5e772e1.png)

There are a couple of interesting observations here.

1. Most neighborhoods' show-up percentages hover near the mean (indicated by the red line) and aligns with the aforementioned 80% show-up rate.
2. One neighborhood has a perfect show up rate.
3. One neghborhood has a zero percent show up rate.

Although the ``Neighbourhood`` variable doesn't appear as if it will have much predictive power due to the show up rate for each neighborhood hovering near the mean show up rate of about 80%, the two neighborhood's show up rates are determined by only a couple observations and are unlikely to hold much predictive power on their own. 

## Correlation Matrix with generated features `DaysTilAppt`

![image](https://user-images.githubusercontent.com/42311832/98758471-25b5d800-238c-11eb-8ec1-4702645d1780.png)

Interestingly, there seems to be a strong correlation between ``AppointmentID`` and ``Appointment``, ``Scheduled`` and ``DaysTilAppt``, indicating that it is not just a randomly generated variable, but somehow captures some information contained in the latter variables. This probably explains why it also has one of the highest correlations with the target variable out of any of the original variables. Since we weren't given any insight into how ``AppointmentID`` was generated in the data documentation, it might also capture information from other sources that are not a part of this dataset. Therefore, we can exclude it as a model feature, yet still capture some of its prediction power from a combination of one or more of the ``Appointment``, ``Scheduled`` and ``DaysTilAppt`` features. We can also see that ``Age`` significantly correlates with both ``Hipertension`` and ``Diabetes``, while ``Hipertensiopn`` and ``Diabetes`` correlate with each other. However, since none of them have a high correlation with our target variable ``is_noshow``, we'll see how much predictive power they ultimately hold.


> ### A quick word on the parameter values: 
The hyperparameter ranges, for the hyperparameters included in the models below, were decided upon by several trials in the build up to the final model, where the highest performing values, or ranges, were honed in upon. You will see these parameters used in the upcoming models (see footnote for hyperparameter definitions)[<sub>1</sub>](#fn1).

![image](https://user-images.githubusercontent.com/42311832/98758700-a70d6a80-238c-11eb-8e04-72717cd5b22c.png)

> Since our data is unbalnced, with and 80/20 split between patients who arrived to their appointments and no-shows, respectively, we first balanced our data with class weights. At first glance, it appears as if our decision tree has a good classification split right from the root node. `DaysTilAppt $<= 0.5` seems to be the rule that best classifies our observations, with same-day appointments being the greatest predictor of a patient's arrival. We can confirm that with the built-in ``feature_importance_`` method for decision tree classifiers, below.

> Although the range of *Days Until Appointment* has been truncated to improve the visibility of the frequency disparity, the distribution plot below shows us that the frequency of no-shows is higher at every *Days Until Appointment* bin level except that representing same day appointments (zero days). We can confirm that the average days between scheduling and appointment is seven days longer for no-shows than it is for *show-ups*.

![image](https://user-images.githubusercontent.com/42311832/98758746-c0161b80-238c-11eb-9340-ca285dab0488.png)

### Feature Importance:

+ Age - 0.109
+ Scholarship - 0.001
+ Hipertension - 0.0
+ Diabetes - 0.0
+ Alcoholism - 0.002
+ Handcap - 0.0
+ SMS_received - 0.011
+ is_male - 0.008
+ TimeOfDay - 0.012
+ Scheduled - 0.004
+ Appointment - 0.003
+ DaysTilAppt - 0.85

## Logistic Regression

```python
# Define our features and target variable
X = no_show.drop(['is_noshow','Neighbourhood','ScheduledDay','AppointmentDay','PatientId','AppointmentID'], axis=1)
Y = no_show.is_noshow
```
Execution time: 53.11 seconds

|     C |  max_iter | class_weight   |   train_precision |   test_precision |   train_recall |   test_recall |   train_acc |   test_acc |   train_f1 |   test_f1 |
| -----:|----------:|:---------------|------------------:|-----------------:|---------------:|--------------:|------------:|-----------:|-----------:|----------:|
|  0.01 |      3000 | balanced       |          0.321798 |         0.31218  |       0.57161  |      0.56527  |    0.667951 |   0.666184 |   0.411778 |  0.402225 |
|  0.01 |      2000 | balanced       |          0.321798 |         0.31218  |       0.57161  |      0.56527  |    0.667951 |   0.666184 |   0.411778 |  0.402225 |
|  0.01 |      3000 | balanced       |          0.321798 |         0.31218  |       0.57161  |      0.56527  |    0.667951 |   0.666184 |   0.411778 |  0.402225 |
|  0.01 |      3000 | balanced       |          0.321798 |         0.31218  |       0.57161  |      0.56527  |    0.667951 |   0.666184 |   0.411778 |  0.402225 |
|  0.25 |      4000 | balanced       |          0.322187 |         0.31136  |       0.572691 |      0.564967 |    0.668144 |   0.66531  |   0.412377 |  0.401467 |
|  0.25 |      2000 | balanced       |          0.322187 |         0.31136  |       0.572691 |      0.564967 |    0.668144 |   0.66531  |   0.412377 |  0.401467 |
|  0.5  |      2000 | balanced       |          0.321427 |         0.311688 |       0.572055 |      0.564663 |    0.667434 |   0.665762 |   0.41159  |  0.401663 |
|  0.5  |      4000 | balanced       |          0.321427 |         0.311688 |       0.572055 |      0.564663 |    0.667434 |   0.665762 |   0.41159  |  0.401663 |
|  1    |      4000 | balanced       |          0.321071 |         0.310792 |       0.572564 |      0.564359 |    0.666917 |   0.664797 |   0.411429 |  0.400841 |
|  1    |      3000 | balanced       |          0.321071 |         0.310792 |       0.572564 |      0.564359 |    0.666917 |   0.664797 |   0.411429 |  0.400841 |

None of the model parameter combinations were able to perform above our previous baseline of 80% accuracy. However, let's say, for instance, that a medical office was more concerned with targeting as many no-shows as possible, even if it means targeting some *show-ups* in the process. Depending on what intervention measures they choose to employ to combat no-shows, as well as the associated costs, the best **recall** (the ratio of predicted positives to actual positives, also known as the **True Positive Rate**) or **f1 score** (the reciprocal of the arithmetic mean of recall and precision)[<sub>2</sub>](#fn2) might better identify the preferred parameters for this model than the accuracy score alone. Several parameter combinations score over 0.5 on ``test_recall`` in our logistic regression model, which means over 50% of all cancellations are being identified.

This model takes some time to run, though. At the moment, we're using all of our categorical besides ``Neighbourhood``, the ID features and the Schedule/Appointment variables from which we've extracted our ``Schedule``, ``Appointment``, ``TimeOfDay`` and ``DaysTilAppt`` features. We've seen previously, from our correlation heatmap, that our target variable only shares a small correlation with a handful of features, anyway. So, let's reduce our model features and see how the model performs.

```python
# Define our features and target variable
X = no_show[['Age','TimeOfDay','SMS_received','DaysTilAppt']]
Y = no_show.is_noshow
```
Execution time: 3.89 seconds

|     C |  max_iter | class_weight   |   train_precision |   test_precision |   train_recall |   test_recall |   train_acc |   test_acc |   train_f1 |   test_f1 |
| -----:|----------:|:---------------|------------------:|-----------------:|---------------:|--------------:|------------:|-----------:|-----------:|----------:|
|  1    |      4000 | balanced       |          0.31992  |         0.309728 |       0.567033 |      0.559654 |    0.666878 |   0.664706 |   0.409052 |  0.398767 |
|  1    |      4000 | balanced       |          0.31992  |         0.309728 |       0.567033 |      0.559654 |    0.666878 |   0.664706 |   0.409052 |  0.398767 |
|  0.5  |      3000 | balanced       |          0.319908 |         0.309644 |       0.567033 |      0.559502 |    0.666865 |   0.664646 |   0.409043 |  0.398659 |
|  0.25 |      3000 | balanced       |          0.31992  |         0.309644 |       0.567033 |      0.559502 |    0.666878 |   0.664646 |   0.409052 |  0.398659 |
|  0.25 |      2000 | balanced       |          0.31992  |         0.309644 |       0.567033 |      0.559502 |    0.666878 |   0.664646 |   0.409052 |  0.398659 |
|  0.5  |      3000 | balanced       |          0.319908 |         0.309644 |       0.567033 |      0.559502 |    0.666865 |   0.664646 |   0.409043 |  0.398659 |
|  0.5  |      3000 | balanced       |          0.319908 |         0.309644 |       0.567033 |      0.559502 |    0.666865 |   0.664646 |   0.409043 |  0.398659 |
|  0.5  |      4000 | balanced       |          0.319908 |         0.309644 |       0.567033 |      0.559502 |    0.666865 |   0.664646 |   0.409043 |  0.398659 |
|  0.01 |      4000 | balanced       |          0.320313 |         0.310232 |       0.566652 |      0.559199 |    0.667408 |   0.6654   |   0.409275 |  0.399068 |
|  0.01 |      3000 | balanced       |          0.320313 |         0.310232 |       0.566652 |      0.559199 |    0.667408 |   0.6654   |   0.409275 |  0.399068 |

With this search we see a slight downtick in ``test_recall`` and ``test_f1`` scores but a significant improvement in execution time despite the runtime on the first search not being all that prohibitive. Nonetheless, let's reduce our features a bit, to just the two highest correlating features, and see what we get.

```python
# Define our features and target variable
X = no_show[['SMS_received','DaysTilAppt']]
Y = no_show.is_noshow
```
Execution time: 3.38 seconds


|     C |  max_iter | class_weight   |   train_precision |   test_precision |   train_recall |   test_recall |   train_acc |   test_acc |   train_f1 |   test_f1 |
| -----:|----------:|:---------------|------------------:|-----------------:|---------------:|--------------:|------------:|-----------:|-----------:|----------:|
|  1    |      4000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |
|  0.01 |      4000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |
|  0.01 |      4000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |
|  0.5  |      4000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |
|  0.01 |      3000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |
|  0.5  |      3000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |
|  0.5  |      4000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |
|  0.25 |      3000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |
|  0.25 |      3000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |
|  0.01 |      3000 | balanced       |          0.301101 |         0.292004 |       0.594431 |      0.590923 |    0.636995 |   0.634066 |   0.399726 |  0.390863 |

Our random search model executed in about the same time with these two features as it did with features from our second search. It also outperformed both of our previous searches in ``test_recall`` score while maintaining only a slight downtick in ``test_f1`` score compared to the original features we modeled.

## Random Forest
Let's run the previous three collections of features on the random forest model.

```python
# Define our features and target variable
X = no_show.drop(['is_noshow','Neighbourhood','ScheduledDay','AppointmentDay','PatientId','AppointmentID'], axis=1)
Y = no_show.is_noshow
```
Execution time: 28.57 seconds

| criterion | max_depth | n_estimators | class_weight | train_precision | test_precision | train_recall | test_recall | train_acc |test_acc | train_f1 | test_f1 |
| :---------|----------:|-------------:|:-------------|----------------:|---------------:|-------------:|------------:|----------:|--------:|---------:|--------:|
|  gini     |         3 |          100 | balanced_subsample |  0.29757  |       0.28958  |     0.86409  |    0.85929  |  0.557634 |0.553213 | 0.442689 |0.433179 |
|  gini     |         5 |           50 | balanced           |  0.300329 |       0.290785 |     0.86962  |    0.85929  |  0.561563 |0.555656 | 0.446468 |0.434526 |
|  entropy  |         5 |           50 | balanced           |  0.301283 |       0.291086 |     0.866061 |    0.855495 |  0.564381 |0.557345 | 0.447048 |0.434374 |
|  gini     |         3 |          250 | balanced           |  0.297075 |       0.289336 |     0.858687 |    0.854129 |  0.558151 |0.554209 | 0.441431 |0.432248 |
|  gini     |         4 |           50 | balanced           |  0.300038 |       0.292581 |     0.857352 |    0.851852 |  0.564316 |0.561356 | 0.444514 |0.435562 |
|  entropy  |         4 |           50 | balanced_subsample |  0.301201 |       0.291968 |     0.8543   |    0.845325 |  0.567379 |0.561989 | 0.445376 |0.434027 |
|  gini     |         5 |           50 | balanced_subsample |  0.304882 |       0.294389 |     0.847181 |    0.834548 |  0.576194 |0.56971  | 0.448396 |0.435244 |
|  gini     |         4 |          100 | balanced           |  0.303543 |       0.29407  |     0.838599 |    0.829539 |  0.575962 |0.570494 | 0.445743 |0.434213 |
|  gini     |         4 |           50 | balanced           |  0.30518  |       0.295343 |     0.834721 |    0.824074 |  0.579981 |0.574414 | 0.446952 |0.434842 |
|  entropy  |         3 |          250 | balanced           |  0.302442 |       0.293312 |     0.823597 |    0.816181 |  0.5779   |0.572786 | 0.442418 |0.431541 |

There are some attractive ``test_recall`` scores among these trials, with each eclipsing 0.82. We're also seeing an improvement in the ``test_f1`` scores. We do see some low precision scores, however, and this is where a medical office's no-show intervention methods and costs would come into play. A doctor's office would have to determine whether the costs of a no-show outweigh the cost of wasting intervention methods on patients who have a high likelihood of showing up. Also, although this model vastly outperforms the logistic model from before in ``test_recall`` score, it takes longer time to run. Now let's see if a reduced predictor set can approximate these improved results while maintaining or improving speed.

```python
# Define our features and target variable
X = no_show[['Age','TimeOfDay','SMS_received','DaysTilAppt']]
Y = no_show.is_noshow
```
Execution time: 49.64 seconds

| criterion | max_depth | n_estimators | class_weight | train_precision | test_precision | train_recall | test_recall | train_acc |test_acc | train_f1 | test_f1 |
| :---------|----------:|-------------:|:-------------|----------------:|---------------:|-------------:|------------:|----------:|--------:|---------:|--------:|
|  entropy  |         3 |          100 | balanced_subsample |  0.290141 |       0.283822 |     0.905283 |    0.904372 |  0.5304   |0.527609 | 0.439442 |0.432052 |
|  entropy  |         3 |           50 | balanced           |  0.29113  |       0.284227 |     0.901723 |    0.899059 |  0.533593 |0.530112 | 0.440152 |0.43191  |
|  gini     |         3 |          100 | balanced           |  0.291493 |       0.284627 |     0.90007  |    0.897086 |  0.534859 |0.53159  | 0.44037  |0.432144 |
|  gini     |         4 |          250 | balanced_subsample |  0.300273 |       0.28983  |     0.86854  |    0.855191 |  0.561744 |0.554902 | 0.446263 |0.432935 |
|  gini     |         4 |           50 | balanced           |  0.300484 |       0.289965 |     0.869112 |    0.853522 |  0.562002 |0.555656 | 0.446571 |0.432871 |
|  gini     |         4 |          100 | balanced_subsample |  0.300765 |       0.290529 |     0.866887 |    0.853066 |  0.563153 |0.556923 | 0.446588 |0.433441 |
|  gini     |         4 |          100 | balanced_subsample |  0.301355 |       0.290625 |     0.865234 |    0.850789 |  0.564743 |0.557767 | 0.447017 |0.433253 |
|  gini     |         5 |          100 | balanced_subsample |  0.304529 |       0.2933   |     0.858242 |    0.842593 |  0.572653 |0.565367 | 0.449546 |0.435134 |
|  gini     |         5 |          100 | balanced_subsample |  0.305501 |       0.294136 |     0.853347 |    0.836825 |  0.575742 |0.568594 | 0.449926 |0.435277 |
|  entropy  |         5 |          250 | balanced_subsample |  0.305685 |       0.293825 |     0.851821 |    0.834244 |  0.576479 |0.568714 | 0.449914 |0.434587 |

```python
# Define our features and target variable
X = no_show.drop(['is_noshow','Neighbourhood','ScheduledDay','AppointmentDay','PatientId','AppointmentID'], axis=1)
Y = no_show.is_noshow
```
Execution time: 36.01 seconds

|criterion | max_depth | n_estimators | class_weight | train_precision | test_precision | train_recall | test_recall | train_acc | test_acc | train_f1 | test_f1 |
| :--------|----------:|-------------:|:-------------|----------------:|---------------:|-------------:|------------:|----------:|---------:|---------:|--------:|
|  entropy |         4 |          250 | balanced           |  0.286859 |       0.281288 |     0.919077 |    0.920461 |  0.518974 | 0.516934 | 0.437247 |0.430896 |
|  gini    |         4 |           50 | balanced_subsample |  0.286859 |       0.281288 |     0.919077 |    0.920461 |  0.518974 | 0.516934 | 0.437247 |0.430896 |
|  entropy |         4 |          250 | balanced           |  0.286859 |       0.281288 |     0.919077 |    0.920461 |  0.518974 | 0.516934 | 0.437247 |0.430896 |
|  entropy |         3 |          250 | balanced_subsample |  0.286859 |       0.281288 |     0.919077 |    0.920461 |  0.518974 | 0.516934 | 0.437247 |0.430896 |
|  gini    |         4 |          250 | balanced           |  0.286859 |       0.281288 |     0.919077 |    0.920461 |  0.518974 | 0.516934 | 0.437247 |0.430896 |
|  gini    |         4 |          100 | balanced_subsample |  0.286859 |       0.281288 |     0.919077 |    0.920461 |  0.518974 | 0.516934 | 0.437247 |0.430896 |
|  entropy |         4 |          100 | balanced_subsample |  0.286859 |       0.281288 |     0.919077 |    0.920461 |  0.518974 | 0.516934 | 0.437247 |0.430896 |
|  entropy |         5 |           50 | balanced           |  0.286916 |       0.28122  |     0.91895  |    0.919854 |  0.519142 | 0.516964 | 0.437299 |0.43075  |
|  entropy |         5 |           50 | balanced           |  0.286973 |       0.281246 |     0.91895  |    0.919854 |  0.519272 | 0.517024 | 0.437365 |0.43078  |
|  entropy |         5 |          100 | balanced_subsample |  0.286971 |       0.281218 |     0.918759 |    0.919551 |  0.519323 | 0.517054 | 0.43734  |0.430715 |

### Cross validation

```python
rfc = ensemble.RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', max_depth=3, 
                                      n_estimators=100)
cvs = cross_val_score(rfc, X, Y, cv=5, scoring='recall')
print(cvs, f', mean score: {np.mean(cvs)}')
```
⮕  [0.93839606 0.95766129 0.86623348 0.91218638 0.92293907] , mean score: 0.9194832541879588

Again, the simplest model outperforms its larger counterparts, consistently testing above 0.91 even at a mere max-depth of three. The low tree depth is a form of regularization, meaning that, along with the consistency between training and testing scores, our model is performing as intended (high recall) without overfitting. You can see from our cross validation scores that if we take the hyperparameters from one of our top performing hyperparameter combinations, we get high recall scores across the board.

## Support Vector Machines
In the interest of saving time, let's only run the support vector classifier model on the minimized, two feature set.

```python
X = no_show[['SMS_received','DaysTilAppt']]
Y = no_show.is_noshow
```
Execution time: 3190.07 seconds

| C	   | class_weight	| kernel | model_class     | test_acc	| test_f1	| test_precision	| test_recall	| train_acc	| train_f1	| train_precision	| train_recall |
|:-----|-------------:|-------:|:----------------|---------:|--------:|----------------:|------------:|----------:|----------:|----------------:|-------------:|
| 0.01 | balanced	    | linear | sklearn.svm.SVC | 0.636931	|0.378504	| 0.286787	      | 0.556466	  | 0.638636	| 0.386967	| 0.295364	      | 0.560931     |
| 0.01 | balanced	    | linear | sklearn.svm.SVC | 0.636931	|0.378504	| 0.286787	      | 0.556466	  | 0.638636	| 0.386967	| 0.295364	      | 0.560931     |
| 0.01 | balanced	    | linear | sklearn.svm.SVC | 0.636931	|0.378504	| 0.286787	      | 0.556466	  | 0.638636	| 0.386967	| 0.295364	      | 0.560931     |
| 0.01 | balanced	    | linear | sklearn.svm.SVC | 0.636931	|0.378504	| 0.286787	      | 0.556466	  | 0.638636	| 0.386967	| 0.295364	      | 0.560931     |
| 0.25 | balanced	    | linear | sklearn.svm.SVC | 0.637896	|0.374863	| 0.285284	      | 0.546448	  | 0.639502	| 0.384237	| 0.294344	      | 0.553175     |
| 0.25 | balanced	    | linear | sklearn.svm.SVC | 0.637896	|0.374863	| 0.285284	      | 0.546448	  | 0.639502	| 0.384237	| 0.294344	      | 0.553175     |
| 0.5	 | balanced	    | linear | sklearn.svm.SVC | 0.63865	|0.373195	| 0.284722	      | 0.541439	  | 0.640278	| 0.382261	| 0.29367	        | 0.547391     |
| 1	   | balanced	    | linear | sklearn.svm.SVC | 0.63865	|0.373195	| 0.284722	      | 0.541439	  | 0.640278	| 0.382261	| 0.29367	        | 0.547391     |
| 0.5	 | balanced	    | linear | sklearn.svm.SVC | 0.63865	|0.373195	| 0.284722	      | 0.541439	  | 0.640278	| 0.382261	| 0.29367	        | 0.547391     |
| 0.5	 | balanced	    | linear | sklearn.svm.SVC | 0.63865	|0.373195	| 0.284722	      | 0.541439	  | 0.640278	| 0.382261	| 0.29367	        | 0.547391     |

This model took an exceedingly long time to run and it is out worst performing model yet. Let's try a gradient boosting model.

## Gradient Boosting

Gradient Boosting builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage ``n_classes_``regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced. An initial run of the model produces `nan` values in some of our f1 score columns, indicating that the model is just making a naive, dominant class classification nearly every time (causing division by zero in the f1 formula). This is likely due to the fact that the gradient boosting algorithm doesn't allow us to balance class weights. Therefore, we will employ **Oversampling**.

## Oversampling

In an attempt to overcome the lack of a 'balance class weights' hyperparameter, we'll oversample our minority class. We'll do so after splitting test and training sets in order to avoid cross-set "bleed" where our testing set informs our training set. Since we've already split our data in the cell above, we'll proceed with the oversampling. We'll be using the SMOTE algorithm (Synthetic Minority Oversampling Technique). Also, we'll reproduce the GBoost_RandomSearch function in order to incorporate SMOTE into the built-in test/train split.

```python
sm = SMOTE(random_state=0, sampling_strategy=1.0)

X = no_show[['SMS_received','DaysTilAppt']]
Y = no_show.is_noshow
```
Execution time: 46.25 seconds

|loss|learning_rate|max_depth|max_features|n_estimators|train_precision| test_precision | train_recall | test_recall | train_acc | test_acc | train_f1 | test_f1 |
| :----|--------------:|----------:|:----------|--------:|-----------:|-------------:|-------------:|------------:|----------:|---------:|----------:|----------:|
|  exponential |  0.01 |         3 | sqrt      |     100 |   0.611708 |     0.281288 |     0.918669 |    0.920461 |  0.667764 | 0.516934 |  0.734404 |  0.430896 |
|  exponential |  0.01 |         4 | sqrt      |     100 |   0.611748 |     0.281254 |     0.918669 |    0.92031  |  0.667813 | 0.516903 |  0.734432 | 0.43084   |
|  deviance    |  0.01 |         4 | sqrt      |     250 |   0.611748 |     0.281254 |     0.918669 |    0.92031  |  0.667813 | 0.516903 |  0.734432 |  0.43084  |
|  exponential |  0.1  |         5 | sqrt      |      50 |   0.612247 |     0.281272 |     0.918474 |    0.918488 |  0.668389 | 0.517507 |  0.734729 |  0.430661 |
|  deviance    |  0.25 |         3 | sqrt      |      50 |   0.612371 |     0.281224 |     0.918442 |    0.918033 |  0.668535 | 0.517537 |  0.734808 |  0.430555 |
|  deviance    |  0.1  |         5 | sqrt      |     100 |   0.612484 |     0.281254 |     0.918393 |    0.917577 |  0.668665 | 0.517748 |  0.734874 |  0.43054  |
|  deviance    |  0.1  |         4 | sqrt      |     100 |   0.61243  |     0.28115  |     0.918442 |    0.917577 |  0.668608 | 0.517507 |  0.734851 |  0.430418 |
|  exponential |  0.5  |         4 | sqrt      |     250 |   0.61305  |     0.281449 |     0.915781 |    0.914238 |  0.668876 | 0.519226 |  0.734443 |  0.430399 |
|  exponential |  0.25 |         3 | sqrt      |     250 |   0.613044 |     0.281435 |     0.915781 |    0.914238 |  0.668868 | 0.519195 |  0.734438 |  0.430383 |
|  exponential |  0.25 |         3 | sqrt      |     250 |   0.613226 |     0.281787 |     0.914402 |    0.913327 |  0.668835 | 0.520281 |  0.734125 |  0.430693 |

The Gradient Boosting Random Search produced `test_recall` scores comparable to the random search algorithm. The model execution times are also very similar. The only notable differences are the cross-validation scores, which are more consistent with Gradient Boosting while scoring almost as high on average (0.918 vs. 0.919).

## Conclusion

Our Random Forest model outperforms any other model across various hyperparameter combinations, only slightly edging out the Gradient Boosting model. Given the algorithm's ability to apply class weights to our unbalanced data set while performing well on both training and testing sets (not to mention at low max depths), we can have a high degree of confidence that our model is not overfitting the training data while simultaneously not making naive classifications. Also, being able to apply the class weights hyperparameter directly, it negates the neccessity of using the SMOTE algorithm to balance our classes. Ultimately, Random Forests is the preferred model over Gradient Boost due to simplicity alone.

### Footnotes

<span id="fn1">
1.
    
1. **C**, our inverse regularization parameter, affects the sensitivity of our output/prediction to our various inputs/features. Since the relationship is inverse, the smaller our 'C' value, the stronger our regularization. Regularization prevents overfitting by making our predictions less sensitive to our features by shrinking the coefficients of those features.
2. **class_weight**, when 'balanced', adjusts weights inversely proportional to the frequencies of each class in the input data, effectively reducing the predictive power of the larger class while boosting the predictive power of the smaller class. When not specified, all class weights would be equal to one. The "balanced_subsample" mode, in our random forest model, is the same as "balanced" except that weights are computed for every tree grown.
3. **max_iter** is the maximum number of iterations allotted for the logistic regression model to converge towards accuracy.
4. **n_estimators** are the number of trees in the random forest model.
5. **max_depth** is the maximum depth of a tree in our random forest.
6. **max_features** is the number of features to consider when looking for the best node split 
6. **criterion** is the the function used to measure the quality of a node split. "gini" measures the Gini impurity, usually intended for continuous variables, and "entropy" measures the information gained from one node to the next, usually meant for discrete variables.
</span>

#To see the full notebook complete with my code where I created my random search algorithms, a zoom-able decision tree and more details regarding my EDA analysis, click <a href="https://github.com/Ikwuakor/Medical-Appointment-No-Shows/blob/main/Predicting%20Medical%20Appointment%20No-Shows.ipynb">here</href>.
