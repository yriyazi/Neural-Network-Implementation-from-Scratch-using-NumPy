train.py file will run the net with parameters specified in config.yml.

test.py file will load the net with parameters specified in config.yml and return the loss and accuracy.


train_BGD.py  run the net with parameters specified in config.yml but all data will be on same bath by other words batch gradient decent.


train_SGD.py run the net with parameters specified in config.yml but each data will be on diffrent bath by other words stochastic gradient decent.


train_validation_test.py is the same as train.py file but in the validation process the test data will fed to net and loss and accuracy of test will be kept as a list.