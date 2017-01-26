import math
import pandas as pd
from sklearn.cross_validation import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def validation_set_poly(random_seeds, degrees, X, y):
    """
    Use the train_test_split method to create a
    training set and a validation set (50% in each)
    using "random_seeds" separate random samplings over
    linear regression models of varying flexibility
    """
    # Loop over each random splitting into a train-test split
    for i in range(1, random_seeds+1):
        # Increase degree of linear regression polynomial order
        for d in range(1, degrees+1):
            # Create the model, split the sets and fit it
            polynomial_features = PolynomialFeatures(
                degree=d, include_bias=False
            )
            linear_regression = LinearRegression()
            model = Pipeline([
                ("polynomial_features", polynomial_features),
                ("linear_regression", linear_regression)
            ])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=i
            )
            model.fit(X_train, y_train)
            # Calculate the test MSE and append to the
            # dictionary of all test curves
            y_pred = model.predict(X_to_be_predicted)
            test_mse = mean_squared_error(y_actual, pd.DataFrame(y_pred))
            return y_pred, test_mse

def k_fold_cross_val_poly(folds, degrees, X, y):
    n = len(X)
    kf = KFold(n, n_folds=folds)
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X.ix[train_index], X.ix[test_index]
        y_train, y_test = y.ix[train_index], y.ix[test_index]
        # Increase degree of linear regression polynomial order
        for d in range(1, degrees+1):
            # Create the model and fit it
            polynomial_features = PolynomialFeatures(
                degree=d, include_bias=False
            )
            linear_regression = LinearRegression()
            model = Pipeline([
                ("polynomial_features", polynomial_features),
                ("linear_regression", linear_regression)
            ])
            model.fit(X_train, y_train)
            # Calculate the test MSE and append to the
            # dictionary of all test curves
            y_pred = model.predict(X_to_be_predicted)
            test_mse = mean_squared_error(y_actual, pd.DataFrame(y_pred))
            return y_pred, test_mse

if __name__ == "__main__":

    data = pd.read_csv('nl_pb_till_dec_2016.csv')
    X = data[["DayOfTheMonth", "Month", "Year", "Weekday", "IsHoliday"]]
    y = data[["Orderlines"]]
    to_be_predicted_data = pd.read_csv('nl_pb_dec_2016.csv')
    X_to_be_predicted = to_be_predicted_data[["DayOfTheMonth", "Month", "Year", "Weekday", "IsHoliday"]]
    y_actual = to_be_predicted_data[["Orderlines"]]
    degrees = 3

    random_seeds = 10
    prediction_1, mse_1 = validation_set_poly(random_seeds, degrees, X, y)


    folds = 10
    prediction_2, mse_2 = k_fold_cross_val_poly(folds, degrees, X, y)

    if math.sqrt(mse_1) > math.sqrt(mse_2):
        print "MSE: " + str(math.sqrt(mse_2))
        pd.DataFrame(prediction_2).to_csv('output.csv')
    else:
        print "MSE: " + str(math.sqrt(mse_1))
        pd.DataFrame(prediction_1).to_csv('output.csv')
