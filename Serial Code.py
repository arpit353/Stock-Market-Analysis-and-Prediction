# Matplotlib for plotting in the notebook
import matplotlib.pyplot as plt

from stocker import Stocker
amazon = Stocker('AMZN')

amazon.plot_stock()

amazon.plot_stock(stats=['Daily Change'])

model, model_data = amazon.create_prophet_model()

model.plot_components(model_data)
plt.show()

amazon.weekly_seasonality=True
model, model_data = amazon.create_prophet_model()

model.plot_components(model_data)
plt.show()

amazon.weekly_seasonality=False

model, model_data = amazon.create_prophet_model(days=90)

amazon.evaluate_prediction()

amazon.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])

amazon.changepoint_prior_validation(start_date='2016-01-04', end_date='2017-01-03', changepoint_priors=[0.001, 0.05, 0.1, 0.2])

amazon.changepoint_prior_validation(start_date='2016-01-04', end_date='2017-01-03', changepoint_priors=[0.15, 0.2, 0.25,0.4, 0.5, 0.6])

amazon.changepoint_prior_scale = 0.5

amazon.evaluate_prediction()

amazon.weekly_seasonality=True

amazon.evaluate_prediction()

amazon.changepoint_prior_scale=0.5
amazon.weekly_seasonality=True

amazon.evaluate_prediction(nshares=1000)

amazon.evaluate_prediction(start_date = '2008-01-03', end_date = '2009-01-05', nshares=1000)
amazon.predict_future(days=10)
amazon.predict_future(days=100)
