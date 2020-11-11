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
