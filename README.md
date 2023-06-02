# airplane_satisfaction

Prediction for airplane passenger satisfaction

Background:

An airline conducts research to find out what factors influence passenger satisfaction.
So it is hoped that in the future it can predict whether passengers are satisfied or dissatisfied based on these factors.

Dataset Overview:

Customer Type: is the type of customer/passenger loyal or disloyal

Class: type of cabin class whether economy, economy plus, or business class.

Gender: gender (male or female).

Age: the age of the passenger.

Type of Travel: the type of trip whether business or personal.

Flight Distance: flight mileage.

Checkin Service: service when checking in with a score of 0 - 5.

Departure Delay in Minutes: departure delay in minutes.

Arrival Delay in Minutes: arrival delay in minutes.

Satisfaction: passenger satisfaction (satisfied or dissatisfied)

Feature 'satisfaction' will be our target variable.


Result:

The best accuracy score was obtained using the XGBoost Machine Learning algorithm model of 0.818 or 81.8%. 
Then by using a deep learning algorithm model or Artificial Neural Network (ANN), namely MLP (Multi Layer Perceptron) with an accuracy score of 0.817 or 81.7%.

Where the score is obtained after doing hyperparameter tuning.

Then finally did ANN modeling using tensorflow and got the best accuracy score not much different from XGboos and MLP accuracy scores.
