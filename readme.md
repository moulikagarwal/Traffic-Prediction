# Traffic Prediction

For the purpose of cleaning the data and to remove the outliers, we have used three different parameters to predict the estimated value of the flows. In the first measure we used Linear Regression to train the model and estimate the value of the flow.
In the second we use the Temporal property of the flow values to get the estimated values.

## Getting Started

Clone this directory on your local machine and use the “clean_one_zone.sh”, bash file to generate the final output predicted file. The python source code takes one
argument (root directory of a zone) as input, and outputs zone_id.flow.txt. Example:
$bash clean_one_zone.sh path/to/zone/3445


### Analytics and Statistical Approaches

####Method 1
In this method we used the linear regression model to predict the flow values. For this approach we divided the total flow values into training and test set. This set consisted vectors which had higher probability and were not considered to be outliers. We trained the model on the basis of flow values of a specific lane and considered the other lanes as features and adjusted our training set and finally
received an accuracy of approximately 75% for our model . Then using the slope and intercept values received from the Linear regression model we predicted the flow values.

####Method 2
In this method we predict the flow values based on the flow value of the same lane at the preceding and succeeding timestamps which are at a time difference of 5 minutes. Then we assign weights to these values based on their probability and then calculate the final flow value using the flow data from succeeding and preceding time intervals.

####Method 3
In the third method we wanted to give preference to the actual flow values and their probabilities calculated from Lab 9.
Ensemble Step
The final flow value is calculated based on the flow predicted from the above method and taking into account their probabilities as weights to calculate the contribution of each predicted flow to the final output.
