#now we will write a locally weighted regression model from scratch
class local_weight_regression:

    #initialising the model
    def __init__(lwr, bandwidth):
        #setting the bandwidth
        lwr.bandwidth = bandwidth 
    #defining the kernel function
    def kernel_func(lwr, distance):
        #gaussian kernel function
        return np.exp(-1 * (distance)**2 / (2 * lwr.bandwidth**2))
    #defining the fit method
    def fit(lwr, X_train, y_train):
        lwr.X_train = X_train.to_numpy()
        lwr.y_train = y_train.to_numpy()
        lwr.global_mean = np.mean(lwr.y_train)
    #defining the predict method
    def predict(lwr, X_test):
        predictions = []
        X_test_np = X_test.to_numpy()

        for i in range(X_test_np.shape[0]):
            #dist calc
            distances = np.linalg.norm(lwr.X_train - X_test_np[i], axis=1)
            
            #weight cal
            weight = lwr.kernel_func(distances)

            #prediction cal
            #explanation of the formula:
            #sum of (weights * y_train) divided by sum of weights
            weight_sum = np.sum(weight)
            if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
                pred = lwr.global_mean
            else:
                pred = np.sum(weight * lwr.y_train) / weight_sum

            predictions.append(pred)

        return np.array(predictions)
    
    
