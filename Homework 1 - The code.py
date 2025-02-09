import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# Linear regression functions :


# Data split function:
def train_test_split(df, test_size_ratio, random_seed = None):
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
    indices = list(range(len(df)))
    np.random.shuffle(indices)
    test_indices = indices[:int(test_size_ratio*len(df))]
    df_test = df.iloc[test_indices]
    df_train = df.drop(index = df.index[test_indices] )
    
    return df_train, df_test





def phi_affine_design(x, k = None):
    
    phi = np.column_stack((np.ones(x.shape[0]), x))
    
    return phi

def phi_polynomial_design(x,k):
    
    phi = np.column_stack([x**i for i in range(k+1)])
    
    return phi  


def plot_2_dim_data(df, df_name):
    
    column_names = df.columns
    plt.scatter(df[column_names[0]], df[column_names[1]]) 
    plt.xlabel(str(column_names[0]))
    plt.ylabel(str(column_names[1]))
    plt.title(str(df_name))
    plt.savefig("plot_"+ str(df_name) +"_" + str(column_names[0]) +"_" + str(column_names[1]) + ".png")
    plt.close()   

def plot_scatter_with_fitted_model(df, df_name, phi_design, design_name, model, theta_optimal, k = None):
    
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    x_range = np.linspace(np.min(x) - 0.05, np.max(x) + 0.05, 100)
    x_range = x_range.reshape(-1,1)
    
    # Predictions of the model for obtained parameters theta:
    phi = phi_design(x_range, k)
    y_model = model(theta_optimal, phi)
    
    #Plotting: 
    column_names = df.columns
    plt.scatter(x, y) 
    plt.plot(x_range, y_model, '-r', label = str(design_name))
    plt.legend()
    if k < 4:
       plt.text(0, np.max(y) - 3 , "theta = " + str(np.round(theta_optimal,2)))
    plt.xlabel(str(column_names[0]))
    plt.ylabel(str(column_names[1]))
    plt.title(str(df_name))
    plt.savefig("plot_"+ str(df_name) +"_with_" + str(design_name) + str(column_names[0]) +"_" + str(column_names[1]) + ".png")
    plt.close()
    
def plot_train_test_error_evolution(df_train, df_test, phi_design, max_polynomial_degree):
    
    polynomial_degrees = np.arange(1,max_polynomial_degree + 1)
    MSE_train_errors = np.array([])
    MSE_test_errors = np.array([])
    
    for k in range(1,max_polynomial_degree + 1):
        
        # 1) We train model for a give polynomial degree k:
        
        phi_polynomial_train = phi_design(x = df_train["x"].to_numpy(), k = k)
        theta_polynomial_optimal = normal_equations(phi = phi_polynomial_train, y = df_train["y"].to_numpy())
        
        
        # 2) We calculate corresponding MSE on train and test datas:
        
        phi_polynomial_test = phi_design(x = df_test["x"].to_numpy(), k = k)
        
        MSE_train_error = calculate_MSE(y = df_train["y"].to_numpy(), phi = phi_polynomial_train, theta = theta_polynomial_optimal)
        MSE_test_error = calculate_MSE(y = df_test["y"].to_numpy(), phi = phi_polynomial_test, theta = theta_polynomial_optimal)
        
        # 3) We add obtained values of MSE's to the corresponding arrays:
        
        MSE_train_errors = np.append(MSE_train_errors, MSE_train_error)
        MSE_test_errors = np.append(MSE_test_errors, MSE_test_error)
        
        plot_scatter_with_fitted_model(df = df, df_name = "Scatter plot with fitted model", phi_design = phi_polynomial_design, design_name = "Polynomial design for k = " +str(k), model = linear_model, theta_optimal = theta_polynomial_optimal, k = k)
    
    # 4) Plotting:  
    plt.scatter(polynomial_degrees, MSE_train_errors, color = 'b', label = "MSE's on train data")   
    plt.scatter(polynomial_degrees, MSE_test_errors, color = 'r', label = "MSE's on test data") 
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title("MSE on Train and Test Data")  
    plt.legend()  
    plt.savefig("Train_and_test_errors.png")
    plt.close()
    
    best_k = polynomial_degrees[np.argmin(MSE_test_errors)]
    print("The best value of k: ", best_k)    
    
    
def calculate_MSE(y,phi,theta):
    
    residuals = y - np.dot(phi, theta)  
    
    return np.mean(residuals**2) 

def linear_model(theta, phi):
    
    return np.dot(phi, theta)



def normal_equations(phi,y):
    
    A = np.linalg.inv(np.dot(phi.T, phi))
    theta = np.dot(np.dot(A,phi.T),y)

    return theta





# KNN functions:


def calculate_distances_for_given_x(x, X):
    
    # The Euclidean metric:
    
    distances = np.sqrt(np.sum((X - x)**2, axis = 1))   
    
    return distances
    
    
def classify_knn(knn_indicies, y):
    class_plus_numbers = 0
    class_minus_numbers = 0
    for knn_index in knn_indicies:
        
        if y[knn_index] == 1:
            class_plus_numbers += 1  
        else:
            class_minus_numbers += 1    
            
    if class_plus_numbers > class_minus_numbers:
        
        return 1
    
    elif class_plus_numbers < class_minus_numbers:
        return -1
    
    else:
        
        return np.random.choice([1, -1])


def calculate_the_number_of_errors(X,y, X_train, y_train, k):
     
     number_of_mistakes_on_data = 0
     probabilities = []
     for x, y in zip(X, y):
        
         # Firt step: we calculate all distances between X_train and given x:
         
         distances = calculate_distances_for_given_x(x , X_train)
         
         # Second step: we sort the objects by increasing distance to them and select the k nearest ones (here we get indicies of objects):
         
         sorted_indices = np.argsort(distances)
         
         knn_indicies = sorted_indices[:k]
         
         # Third step: we obtain a class of a given x:
         
         classification_result_for_x = classify_knn(knn_indicies = knn_indicies, y = y_train)
         
         # Additional step: here we calculated the "probability":
         
         nearest_neighbors_labels = y_train[knn_indicies]
         
         positive_ratio = np.mean(nearest_neighbors_labels == 1)  
         probabilities.append(positive_ratio)
         
         # Fourth step: comparing with real label:
         
         if classification_result_for_x != y:
             number_of_mistakes_on_data += 1
         
     return number_of_mistakes_on_data, np.array(probabilities)       

    
def KNN_algorithm(df_train, df_test, k):
    
    X_train = df_train.drop(columns = "label").to_numpy()
    y_train = df_train["label"].to_numpy()
    
    X_test = df_test.drop(columns = "label").to_numpy()
    y_test = df_test["label"].to_numpy()
    
    # Number of errors for train data:
    
    number_of_errors_on_train_data, probabilities_train = calculate_the_number_of_errors(X = X_train, y = y_train, X_train = X_train, y_train = y_train, k = k)
    
    # Number of errors for test data:
    
    number_of_errors_on_test_data, probabilities_test = calculate_the_number_of_errors(X = X_test, y = y_test, X_train = X_train, y_train = y_train, k = k)     
    
   
    
    
    return number_of_errors_on_train_data, number_of_errors_on_test_data, probabilities_test

def plot_number_of_errors_in_function_of_k(k_values, df_train, df_test):
    
    number_of_errors_on_train_data_values = np.array([])
    number_of_errors_on_test_data_values = np.array([])

    for k_value in k_values:
        
        number_of_errors_on_train_data_value, number_of_errors_on_test_data_value, probabilities_test = KNN_algorithm(df_train = df_train, df_test = df_test, k = k_value)
        
        
        number_of_errors_on_train_data_values = np.append(number_of_errors_on_train_data_values, number_of_errors_on_train_data_value)
        number_of_errors_on_test_data_values = np.append(number_of_errors_on_test_data_values, number_of_errors_on_test_data_value)


    plt.plot(k_values, number_of_errors_on_train_data_values, color = "g", label = "Train Data")
    plt.plot(k_values, number_of_errors_on_test_data_values, color = "r", label = "Test Data")
    plt.title("Number of errors as a function of k")
    plt.xlabel("k number")
    plt.ylabel("Number of errors")
    plt.legend()
    plt.savefig("Number of errors as a function of k.png")
    plt.close()
         
def cross_validation_algorithm_knn(df, number_of_folds, k_values):
    
    size_of_data_i = df.shape[0] // number_of_folds

    average_number_of_errors_on_test_data_values = np.array([])
    
    for k_value in k_values:
        
        test_errors = np.array([])
        for i in range(number_of_folds):
            
            start_index_i = int(i * size_of_data_i)
            stop_index_i = int((i + 1) * size_of_data_i)
            test_indices = range(start_index_i, stop_index_i)

            df_test_i = df.iloc[test_indices]

            df_train_i = df.drop(df.index[test_indices])
            

            
            number_of_errors_on_train_data, number_of_errors_on_test_data_value, probabilities_test = KNN_algorithm(df_train = df_train_i, df_test = df_test_i, k = k_value)      
            
            test_errors = np.append(test_errors, number_of_errors_on_test_data_value)
            
        average_number_of_errors_on_test_data_value = np.mean(test_errors)
        average_number_of_errors_on_test_data_values = np.append(average_number_of_errors_on_test_data_values, average_number_of_errors_on_test_data_value)
        
    
    plt.plot(k_values, average_number_of_errors_on_test_data_values, color = "r", label = "Test Data")
    plt.title("Average number of errors as a function of k")
    plt.xlabel("k number")
    plt.ylabel("Number of errors")
    plt.legend()
    plt.savefig("Average number of errors on test data (crossvalidation) on as a function of k.png")
    plt.close()
    
    best_k_index = np.argmin(average_number_of_errors_on_test_data_values)
    best_k_value = k_values[best_k_index]
    return best_k_value

def create_a_heatmap_for_the_knn_model(df, df_train, k):
    
    # Define a meshgrid:
    
    x_min, x_max = df['x_1'].min() - 1, df['x_1'].max() + 1
    y_min, y_max = df['x_2'].min() - 1, df['x_2'].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

    grid_points = np.c_[xx.ravel(), yy.ravel()] 
    grid_predictions = np.array([
    classify_knn(knn_indicies=np.argsort(calculate_distances_for_given_x(x = point, X = df_train.drop(columns="label").to_numpy()))[:k], y=df_train['label'].to_numpy()) for point in grid_points]).reshape(xx.shape)
    
    plt.contourf(xx, yy, grid_predictions, alpha=0.7, cmap='coolwarm')
    plt.scatter(df['x_1'], df['x_2'], c=df['label'], edgecolor='k', cmap='coolwarm')
    plt.title("Heat map for KNN model for k =" + str(k))
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.savefig("Heat_map_for_the_model.png")
    plt.close()

















# Logistic regression functions:


# Model function, log-loss function and derivative of log-loss function (and minus verions of it): 

def logistic_func(theta, object_features):

    arg = np.dot(object_features, theta)
    arg = np.where(arg > 18, 18, np.where(arg < -18, -18, arg))
    
    return 1.0/(1+np.exp(-arg))

def log_likelihood(theta, object_features, y_train, model, lambda_coef):
    
    norm_theta = np.linalg.norm(theta)
    if norm_theta > 1e6:
       print(f"Warning: Theta norm too large: {norm_theta}. Regularizing heavily.")
       norm_theta = 1e6
    
    epsilon = 1e-10
    y_model_predictions = model(theta, object_features)
    result = np.sum(y_train*np.log(y_model_predictions+epsilon) + (1-y_train)*np.log(1-y_model_predictions + epsilon)) - (lambda_coef/2) * (np.linalg.norm(theta))**2
    
    return result


def negative_log_likelihood(theta, object_features, y_train, model, lambda_coef):
    
    return -1*log_likelihood(theta, object_features, y_train, model, lambda_coef)


def log_likelihood_derivative(theta, object_features, y_train, model, lambda_coef):
    
    y_model_predictions = model(theta, object_features)
    
    delta_y = y_train - y_model_predictions
    
    result = np.dot(object_features.T, delta_y) - lambda_coef*theta
    
    assert result.shape == theta.shape
    
    return result

def negative_log_likelihood_derivative(theta, object_features, y_train, model, lambda_coef):
    
    return -1*log_likelihood_derivative(theta, object_features, y_train, model, lambda_coef)


# Feature designs functions:

def phi_affine_features(X, polynomial_degree = None):
    
    affine_features = np.column_stack((np.ones(X.shape[0]), X))
    
    return affine_features


def phi_2_dim_polynomial_features(X,polynomial_degree):
    
    polynomial_features = []
    
    for a in range(0, polynomial_degree + 1):
        for b in range(0, polynomial_degree + 1 - a):
            polynomial_features.append(X[:,0]**a * X[:,1]**b)
    
    polynomial_features = np.column_stack(polynomial_features)        
            
    
    return polynomial_features


# Normalization methods for data: 

def normalize_data(X, X_means, X_sigmas):
    X_normalized = np.copy(X)
    X_normalized[:, 1:] = (X[:, 1:] - X_means) / X_sigmas
    return X_normalized


def denormalize_theta(theta_normalized, X_means, X_sigmas):
    
     intercept = theta_normalized[0]  
     coefficients = theta_normalized[1:]  
    
    # Coefficients recovering:
     original_coefficients = coefficients / X_sigmas
    
    # Intercept recovering:
     original_intercept = intercept - np.sum((coefficients * X_means) / X_sigmas)
    
    
     theta_original = np.concatenate([[original_intercept], original_coefficients])
     
     return theta_original

# Methods of optimization: 

def batch_gradient_descent(batch_size, object_features_train, y_train, features_design_function_name, init_theta, alpha, nIter, model, lambda_coef, polynomial_degree = None):
    
    epsilon = 1e-6
    n_samples = object_features_train.shape[0]
    theta_iter = init_theta.copy()
    negative_log_likelihood_function_values = []
    
    for iteration in range(nIter):
        
        negative_log_likelihood_function_values.append(negative_log_likelihood(theta = theta_iter, object_features = object_features_train, y_train = y_train, model = logistic_func, lambda_coef = lambda_coef))
        
        current_alpha = alpha / (1 + iteration / nIter)
        
        #Choosing the random batch from doata:
        
        indices = np.random.choice(n_samples, size=batch_size, replace=False)
        
        # Extracting the features and labels of objects in the batch:
        
        batch_features = object_features_train[indices]
        
        batch_labels = y_train[indices]
        
        grad = negative_log_likelihood_derivative(theta = theta_iter, object_features = batch_features, y_train = batch_labels, model = model, lambda_coef = lambda_coef)
        
        if iteration > 1 and abs(negative_log_likelihood_function_values[-1] - negative_log_likelihood_function_values[-2]) < epsilon and np.linalg.norm(grad) < epsilon:
            print(f"Converged after {iteration} iterations with delta < {epsilon}")
            break
        
        theta_iter = theta_iter - current_alpha *  grad
        
    plt.plot(negative_log_likelihood_function_values)
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    if features_design_function_name == "polynomial":
        plt.title("Loss function as a function of iteration for " + str(features_design_function_name) + " features" + " for k = " + str(polynomial_degree))
        plt.savefig("Loss_function_as_a_function_of_iterations_for_" + str(features_design_function_name) + "_features" + "_for_k_" + str(polynomial_degree))
        plt.close()
    else:
        plt.title("Loss function as a function of iterations for " + str(features_design_function_name) + "_features")
        plt.savefig("Loss_function_as_a_function_of_iterations_for_" + str(features_design_function_name) + "_features.png")
        plt.close() 
      
    return theta_iter




def cross_validation_algorithm(
    df, number_of_folds, lambda_coef_values, batch_size, alpha, nIter,
    features_design_function, features_design_function_name, model, polynomial_degree=None
):
    size_of_data_i = df.shape[0] // number_of_folds

    average_number_of_errors_on_test_data_values = np.array([])

    for lambda_coef_value in lambda_coef_values:
        
        test_errors = np.array([])

        for i in range(number_of_folds):
            
            # Split into training and testing data for the current fold
            start_index_i = i * size_of_data_i
            stop_index_i = (i + 1) * size_of_data_i
            test_indices = range(start_index_i, stop_index_i)

            df_test_i = df.iloc[test_indices]
            df_train_i = df.drop(df.index[test_indices])

            # Convert to numpy arrays
            X_train_i = df_train_i.drop('label', axis=1).to_numpy()
            y_train_i = df_train_i['label'].to_numpy()
            X_test_i = df_test_i.drop('label', axis=1).to_numpy()

            # Feature transformation for train set
            object_features_train_i = features_design_function(X=X_train_i, polynomial_degree=polynomial_degree)

            # Compute normalization parameters
            object_features_means = np.mean(object_features_train_i[:, 1:], axis=0)
            object_features_sigmas = np.std(object_features_train_i[:, 1:], axis=0)
            object_features_sigmas[object_features_sigmas < 1e-10] = 1

            # Normalize train and test sets
            object_features_train_i = normalize_data(object_features_train_i, object_features_means, object_features_sigmas)
            object_features_test_i = features_design_function(X=X_test_i, polynomial_degree=polynomial_degree)
            object_features_test_i = normalize_data(object_features_test_i, object_features_means, object_features_sigmas)

            # Initialize theta
            init_theta_i = np.zeros(object_features_train_i.shape[1])

            
            # Optimize theta using gradient descent
            theta_optimal_i = batch_gradient_descent(
                batch_size=batch_size,
                object_features_train=object_features_train_i,
                features_design_function_name=str(features_design_function_name),
                y_train=y_train_i,
                init_theta=init_theta_i,
                alpha=alpha,
                nIter=nIter,
                model=logistic_func,
                lambda_coef=lambda_coef_value,
                polynomial_degree=polynomial_degree
            )

            # Calculate test errors
            test_error_value = calculate_test_errors_for_given_model(
                df_test=df_test_i,
                features_design=features_design_function,
                theta_optimal=theta_optimal_i,
                model=model,
                polynomial_degree=polynomial_degree
            )

            test_errors = np.append(test_errors, test_error_value)

        # Average errors for the current lambda
        average_error = np.mean(test_errors)
        average_number_of_errors_on_test_data_values = np.append(average_number_of_errors_on_test_data_values, average_error)
        
    
    plt.plot(lambda_coef_values, average_number_of_errors_on_test_data_values, color = "r", label = "Test Data")
    plt.title("Number of errors as a function of lambda")
    plt.xlabel("Lambda coefficient")
    plt.ylabel("Number of errors")
    plt.legend()
    plt.savefig("Average number of errors on test data (crossvalidation) on as a function of lambda.png")
    plt.close()
    
    best_lambda = lambda_coef_values[np.argmin(average_number_of_errors_on_test_data_values)]
    
    return best_lambda   

def newton_method(object_features_train, y_train, features_design_function_name, init_theta, nIter, model, lambda_coef, polynomial_degree=None):
    epsilon = 1e-6  
    theta_iter = init_theta.copy()
    negative_log_likelihood_values = []
    
    for iteration in range(nIter):
        # Model predictions
        y_model = model(theta_iter, object_features_train)
        
        # Gradient for -log-likelihood:
        gradient = negative_log_likelihood_derivative(theta = theta_iter, object_features = object_features_train, y_train = y_train, model = logistic_func, lambda_coef = lambda_coef) 
        
        # Hessian:
        R = np.diag(y_model * (1 - y_model))
        hessian = np.dot(object_features_train.T, np.dot(R, object_features_train)) + lambda_coef * np.eye(object_features_train.shape[1])
        
        # Parameters update
        try:
            hessian_inv = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("Error, Hessian matrix couldn't be inverted")
            break
        
        delta_theta = np.dot(hessian_inv, gradient)
        theta_iter -= delta_theta  # Update step for minimization
        
        # Compute -log-likelihood and save it
        negative_log_likelihood_values.append(negative_log_likelihood(theta_iter, object_features_train, y_train, model, lambda_coef))
        
        # Convergence check
        if np.linalg.norm(delta_theta) < epsilon:
            print(f"Newton method converged on iteration number: {iteration + 1}")
            break
    
    # Plot -log-likelihood
    plt.plot(negative_log_likelihood_values)
    plt.xlabel("Iteration")
    plt.ylabel("-Log-likelihood")
    
    if features_design_function_name == "polynomial":
        plt.title("Loss function in Newton's method for " + str(features_design_function_name) + " features" + " k = " + str(polynomial_degree))
        plt.savefig("Loss_function_in_Newton's_method_for_" + str(features_design_function_name) + "_features_k_" + str(polynomial_degree) + ".png")
        plt.close()
    else:
        plt.title("Loss function in Newton's method for " + str(features_design_function_name) + "_features")
        plt.savefig("Loss_function_in_Newton's_method_for_" + str(features_design_function_name) + "_features.png")
        plt.close()
    
    return theta_iter


def hybrid_newton_batch_gradient_descent(batch_size, object_features_train, y_train, features_design_function_name, init_theta, alpha, nIter, model, lambda_coef, polynomial_degree = None):
    

    epsilon = 1e-6  
    n_samples = object_features_train.shape[0]
    theta_iter = init_theta.copy()
    negative_log_likelihood_values = []

    for iteration in range(nIter):
        
        negative_log_likelihood_values.append(negative_log_likelihood(theta_iter, object_features_train, y_train, model, lambda_coef = lambda_coef))

        
        indices = np.random.choice(n_samples, size=batch_size, replace=False)
        batch_features_train = object_features_train[indices]
        batch_labels_train = y_train[indices]

        
        y_model = model(theta_iter, batch_features_train)
        
        
        gradient = negative_log_likelihood_derivative(theta = theta_iter, object_features = batch_features_train, y_train = batch_labels_train, model = logistic_func, lambda_coef = lambda_coef)
        
        R = np.diag(y_model * (1 - y_model))  
        H_approx = np.dot(batch_features_train.T, np.dot(R, batch_features_train))
        H_diag_inv = np.diag(1 / np.diag(H_approx + epsilon))  

        
        delta_theta = np.dot(H_diag_inv, gradient)

        
        theta_iter = theta_iter - alpha * delta_theta

        
        if np.linalg.norm(delta_theta) < epsilon:
            print(f"Converged on iteration number: {iteration + 1}")
            break

    # Visualisation:
    plt.plot(negative_log_likelihood_values)
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    if features_design_function_name == "polynomial":
        plt.title("Loss function in Newton's-batch method for " + str(features_design_function_name) + " features" + " k = " + str(polynomial_degree))
        plt.savefig("Loss_function_in_Newton's-batch_method_for" + str(features_design_function_name) + "_features" + "_k_" + str(polynomial_degree))
        plt.close()
    else:
        plt.title("Loss function in Newton's-batch method for " + str(features_design_function_name) + "_features")
        plt.savefig("Loss_function_in_Newton's-batch_method_for" + str(features_design_function_name) + "_features.png")
        plt.close()

    return theta_iter

# Generalization of optimization methods: 

def optimize(method, object_features_train, y_train, features_design_function_name, init_theta, alpha, nIter, lambda_coef, model, batch_size = None, polynomial_degree = None):
    
    if method == "batch_gradient_descent":
        return batch_gradient_descent(
            batch_size = batch_size,
            object_features_train = object_features_train,
            y_train = y_train,
            features_design_function_name = str(features_design_function_name),
            init_theta = init_theta,
            alpha = alpha,
            nIter = nIter,
            model = model,
            lambda_coef = lambda_coef,
            polynomial_degree = polynomial_degree
        )
    elif method == "newton_method":
        return newton_method(
            object_features_train = object_features_train,
            y_train = y_train,
            features_design_function_name = str(features_design_function_name),
            init_theta = init_theta,
            nIter = nIter,
            model = model,
            lambda_coef = lambda_coef,
            polynomial_degree = polynomial_degree
        )
    elif method == "hybrid_newton_batch_gradient_descent":
        return hybrid_newton_batch_gradient_descent(
            batch_size = batch_size,
            object_features_train = object_features_train,
            y_train = y_train,
            features_design_function_name = str(features_design_function_name),
            init_theta = init_theta,
            alpha = alpha,
            nIter = nIter,
            model = model,
            lambda_coef = lambda_coef,
            polynomial_degree = polynomial_degree
        )
        
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def optimize_with_pipeline(methods_sequence, df, df_train, features_design_function, features_design_function_name, number_of_folds, alpha, nIter, lambda_coef_values, model, batch_size , polynomial_degree = None):
    
    
    print('Data preparation processing...')
    
    # Step 1: Normalize features
    
    X = df.drop('label', axis=1).to_numpy()
    y = df['label'].to_numpy()

    X_train = df_train.drop('label', axis=1).to_numpy()
    y_train = df_train['label'].to_numpy()

    
    object_features = features_design_function(X, polynomial_degree = polynomial_degree)
    object_features_train = features_design_function(X_train, polynomial_degree = polynomial_degree)
    
    
    
    object_features_means = np.mean(object_features[:, 1:], axis=0)
    object_features_sigmas = np.std(object_features[:, 1:], axis=0)
    
    object_features_sigmas[object_features_sigmas < 1e-10] = 1

    # Normalization of object-features matrix:
 
    object_features_train_normalized = normalize_data(object_features_train, object_features_means, object_features_sigmas)
    
    
    print("Cross-validation starts...")
    # Step 2: Cross-validation to find the best lambda
    best_lambda = cross_validation_algorithm(
        df = df_train,
        number_of_folds = number_of_folds,
        lambda_coef_values = lambda_coef_values,
        batch_size = batch_size,
        alpha = alpha,
        nIter = nIter,
        features_design_function = features_design_function,
        features_design_function_name = str(features_design_function_name),
        model = logistic_func,
        polynomial_degree = polynomial_degree
    )
    print("Cross-validation is finished.")
    print(f"Best lambda: {best_lambda}")

    # Step 3: Initialize theta
    n_features = object_features_train_normalized.shape[1]
    init_theta = np.zeros(n_features)
    
    
    theta = init_theta
    for method in methods_sequence:
        print(f"Applying method: {method}")
        theta = optimize(
            method=method,
            object_features_train=object_features_train_normalized,
            y_train=y_train,
            features_design_function_name=features_design_function_name,
            init_theta=theta,
            alpha=alpha,
            nIter=nIter,
            lambda_coef=best_lambda,
            model=model,
            batch_size=batch_size,
            polynomial_degree=polynomial_degree
        )

    print("Optimization complete.")
    
    theta = denormalize_theta(theta_normalized = theta, X_means = object_features_means, X_sigmas = object_features_sigmas)
    
    return theta
    





# Classification and assessment of classification quality methods:

def classify(theta, object_features, model):
    
    model_results = model(theta, object_features)
    
    classifications = np.where(model_results > 0.5, 1, -1)
    
    return classifications

def calculate_train_test_errors_for_given_model(df_train, df_test, features_design, theta_optimal, model, polynomial_degree):
    
    X_train, X_test = df_train.drop('label', axis = 1).to_numpy(), df_test.drop('label', axis = 1).to_numpy()
    y_train , y_test = df_train['label'].to_numpy(), df_test['label'].to_numpy()
    object_features_train = features_design(X = X_train, polynomial_degree = polynomial_degree)
    object_features_test = features_design(X = X_test, polynomial_degree = polynomial_degree)  
    classification_for_train_data_results = classify(theta = theta_optimal, object_features = object_features_train, model = model)
    classification_for_test_data_results = classify(theta = theta_optimal, object_features = object_features_test, model = model)
    train_errors = np.sum(classification_for_train_data_results != y_train)
    test_errors = np.sum(classification_for_test_data_results != y_test)      
    
    return train_errors, test_errors   

def calculate_test_errors_for_given_model(df_test, features_design, theta_optimal, model, polynomial_degree):
    
    X_test = df_test.drop('label', axis = 1).to_numpy()
    y_test = df_test['label'].to_numpy()
    object_features_test = features_design(X = X_test, polynomial_degree = polynomial_degree)  
    classification_for_test_data_results = classify(theta = theta_optimal, object_features = object_features_test, model = model)
    test_errors = np.sum(classification_for_test_data_results != y_test)      
    
    return test_errors  

def numerical_integration_auc(fpr_values, tpr_values):
    x = np.array(fpr_values)
    y = np.array(tpr_values)
    dx = x[1:] - x[:-1]
    
    avg_y = (y[1:] + y[:-1]) / 2
    auc = np.sum(dx * avg_y)
    return auc

def plot_roc_curve(y_true, y_model, data_set_name):
    threshold_values = np.sort(np.unique(y_model))[::-1]  # Thresholds for classification in descending order.
    tpr_values = np.array([])
    fpr_values = np.array([])
    
    for threshold_value in threshold_values:
        y_model_threshold = (y_model >= threshold_value).astype(int)  # Classify with the current threshold.
        
        TP = np.sum((y_true == 1) & (y_model_threshold == 1))
        FN = np.sum((y_true == 1) & (y_model_threshold == 0))
        FP = np.sum((y_true == -1) & (y_model_threshold == 1))
        TN = np.sum((y_true == -1) & (y_model_threshold == 0))
        
        
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        tpr_values = np.append(tpr_values, tpr)
        fpr_values = np.append(fpr_values, fpr)
    
    # Compute AUC using numerical integration
    auc = numerical_integration_auc(fpr_values, tpr_values)
    
    # Plot the ROC curve
    plt.plot(fpr_values, tpr_values, label=f"Classifier {data_set_name} (AUC = {np.round(auc, 2)})")
    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')  # Diagonal line for random classifier.
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    print(f"AUC for {data_set_name}: {auc:.2f}")
    plt.savefig(f"AUC_curve_for_classifier_with_{data_set_name}_features.png")
    plt.close()
    
    return auc


# Function which creates a heatmap of the obtained model:

def create_a_heatmap_for_the_given_model(df, model, theta_optimal, design_features_function, design_features_name, polynomial_degree = None):
    
    # Define a meshgrid:
    
    x_min, x_max = df['x_1'].min() - 1, df['x_1'].max() + 1
    y_min, y_max = df['x_2'].min() - 1, df['x_2'].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

    grid_points = np.c_[xx.ravel(), yy.ravel()] 
    grid_predictions = classify(theta=theta_optimal, object_features = design_features_function(X = grid_points ,polynomial_degree = polynomial_degree ), model=model).reshape(xx.shape)
    
    plt.contourf(xx, yy, grid_predictions, alpha=0.7, cmap='coolwarm')
    plt.scatter(df['x_1'], df['x_2'], c=df['label'], edgecolor='k', cmap='coolwarm')
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    if design_features_name == "polynomial":
        plt.title("Logistic model with " + str(design_features_name) + " features" + " for k = " + str(polynomial_degree))
        plt.savefig("Heat_map_for_the_logistic_model_" +"with_" + str(design_features_name) + "_features_for_k_" +str(polynomial_degree) +".png")
        plt.close()
    else:
        plt.title("Logistic model with " + str(design_features_name) + "_features")
        plt.savefig("Heat_map_for_the_logistic_model_" +"with_" + str(design_features_name) + "_features.png")
        plt.close()

def get_final_results(polynomial_degree_values, train_errors_values, test_errors_values, auc_values):
    
    # Normalazing errors:
    max_errors = max(max(train_errors_values), max(test_errors_values))
    normalized_train_errors = train_errors_values / max_errors
    normalized_test_errors = test_errors_values / max_errors
    
    best_degree_index = np.argmin(test_errors_values)
    best_degree = polynomial_degree_values[best_degree_index]
    
    plt.figure(figsize=(10, 6))

    #Normalized errors plot + AUC plot:
    plt.plot(polynomial_degree_values, normalized_train_errors, label="Train Errors (normalized)", color="green", linestyle="--", marker="o")
    plt.plot(polynomial_degree_values, normalized_test_errors, label="Test Errors (normalized)", color="red", linestyle="--", marker="o")

    # AUC
    plt.plot(polynomial_degree_values, auc_values, label="AUC", color="blue", linestyle="-", marker="s")
    plt.axvline(x = best_degree, color="orange", linestyle="--", label=f"Best Degree ({best_degree})")
    plt.title("Final results")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Normalized Metrics")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("Final_results.png")
    plt.close()
        









# Task 1 - linear regression: 

# Data production:

#Number of points:

n = 40

# x - from uniform distribution on [0,1]:

x = np.random.uniform(0,1,n)


# For y: with a normal distributed noise N(0,1)

y = np.exp(3*x) + np.random.normal(loc = 0, scale = 1, size = n)

data = np.column_stack((x,y))

# Creating of a dataframe:

df = pd.DataFrame(data = data, columns = ["x", "y"])

# Data plot:

plot_2_dim_data(df = df, df_name = "Data plot")

# Data split on a training and test data sets:

df_train, df_test = train_test_split(df, test_size_ratio = 0.2)

# Train data plot:

plot_2_dim_data(df = df_train, df_name = "df_train")

# Test data plot:

plot_2_dim_data(df = df_test, df_name = "df_test")




# Learning of model: 1) affine 2) polynomial



# 1) Obtaining corresponding design marticies:

#For affine version:

phi_affine = phi_affine_design(x = df_train["x"].to_numpy())


# Finding optimal parameters theta:

theta_affine_optimal = normal_equations(phi = phi_affine , y = df_train["y"].to_numpy())



# For polynomial version:

max_polynomial_degree = 9
# Plotting: (fitted model + data)

#plot_scatter_with_fitted_model(df = df, df_name = "Scatter plot with fitted model", phi_design = phi_affine_design, design_name = "Affine design", model = linear_model, theta_optimal = theta_affine_optimal)

# Plotting train and test error as a function of the polynom degree:

plot_train_test_error_evolution(df_train = df_train, df_test = df_test, phi_design = phi_polynomial_design, max_polynomial_degree = max_polynomial_degree)      
        
        
               
# Task 2 - KNN:        
# Data production:

seed = 42
np.random.seed(seed)
n = 200

p = 0.5

mu_plus_1 = np.array([1, 1])
mu_minus1 = np.array([-1, 3])

Sigma_plus_1 = np.array([[2, 1], [1, 1]])
Sigma_minus1 = np.array([[1, -0.5], [-0.5, 1.25]])

y = np.random.choice([-1, 1], size=n, p=[1-p, p])         #Bernoulli

x = np.array([np.random.multivariate_normal(mu_plus_1 if label == 1 else mu_minus1, Sigma_plus_1 if label == 1 else Sigma_minus1) for label in y])

data = np.column_stack((x,y))

df = pd.DataFrame(data = data, columns = ["x_1", "x_2", "label"])

# Data split on a training and test data sets:

df_train, df_test = train_test_split(df, test_size_ratio = 0.5, random_seed = seed)

sns.jointplot(df, x = "x_1", y = 'x_2', hue = 'label')
plt.savefig("Data_Plot_Task_2.png")
plt.close()


# Let's plot the average number of errors in KNN algorithm as a function of k:

k_max = 20

k_values = np.arange(1,k_max + 1)

plot_number_of_errors_in_function_of_k(k_values = k_values, df_train = df_train, df_test = df_test)


# Let's use cross-validation algorithm to test our model: also we find the best k value for this model

best_k_value = cross_validation_algorithm_knn(df = df_train, number_of_folds = 5, k_values = k_values)

print("The best k value (number of neighbors): " , best_k_value )

# Let's plot the heatmap of our model for the best value of k:

create_a_heatmap_for_the_knn_model(df = df, df_train = df_train, k = best_k_value)

# Model analysis for the best k value:

number_of_errors_on_train_data, number_of_errors_on_test_data, probabilities_test = KNN_algorithm(df_train = df_train, df_test = df_test, k = best_k_value)

print("Number of errors on train data: ", number_of_errors_on_train_data)
print("Number of errors on test data: ", number_of_errors_on_test_data)

plot_roc_curve(y_true = df_test['label'].to_numpy(), y_model = probabilities_test, data_set_name = "KNN algorithm" )        
        
# Task 2 - logistic regression:         
        
# Data production:

np.random.seed(42)

n = 200

p = 0.5

mu_plus_1 = np.array([1, 1])
mu_minus1 = np.array([-1, 3])

Sigma_plus_1 = np.array([[2, 1], [1, 1]])
Sigma_minus1 = np.array([[1, -0.5], [-0.5, 1.25]])

y = np.random.choice([-1, 1], size=n, p=[1-p, p])         #Bernoulli

x = np.array([np.random.multivariate_normal(mu_plus_1 if label == 1 else mu_minus1, Sigma_plus_1 if label == 1 else Sigma_minus1) for label in y])

data = np.column_stack((x,y))

df = pd.DataFrame(data = data, columns = ["x_1", "x_2", "label"])

# Data split on a training and test data sets:

df_train, df_test = train_test_split(df = df, test_size_ratio = 0.5, random_seed = 42)

# Let's obtain optimal parameters for affine features and create a heatmap:


# Parameters for gradient descent:

alpha = 0.0001

nIter = 10000

lambda_coef_values = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

batch_fraction = 0.2
batch_size = int(batch_fraction * df_train.shape[0])                  

methods_sequence = ["batch_gradient_descent", "hybrid_newton_batch_gradient_descent", "newton_method"]


# Let's obtain resulst for the polynomial features (k = 1,2,3,4,5): k = 1 - affine model

max_polynomial_degree = 5

polynomial_degree_values = np.arange(1,max_polynomial_degree + 1)

train_errors_values = []
test_errors_values = []
auc_values = []
theta_values = []
for polynomial_degree in polynomial_degree_values:
    
    print("polynomial_degree: ", polynomial_degree)

    theta_opt_for_polynomial_features = optimize_with_pipeline(methods_sequence = methods_sequence,
                                                   df = df, 
                                                   df_train = df_train, 
                                                   features_design_function = phi_2_dim_polynomial_features, 
                                                   features_design_function_name = "polynomial",
                                                   number_of_folds = 5,
                                                   alpha = alpha,
                                                   nIter = nIter,
                                                   lambda_coef_values = lambda_coef_values,
                                                   model = logistic_func,
                                                   batch_size = batch_size,
                                                   polynomial_degree = polynomial_degree
                                                   ) 
    print("theta :", theta_opt_for_polynomial_features)
    create_a_heatmap_for_the_given_model(df = df, model = logistic_func, theta_optimal = theta_opt_for_polynomial_features, design_features_function = phi_2_dim_polynomial_features, design_features_name = "polynomial", polynomial_degree = polynomial_degree)
    
    train_errors_value, test_errors_value = calculate_train_test_errors_for_given_model(df_train = df_train, df_test = df_test, features_design = phi_2_dim_polynomial_features, theta_optimal = theta_opt_for_polynomial_features, model = logistic_func, polynomial_degree = polynomial_degree)
    
    train_errors_values.append(train_errors_value)
    
    test_errors_values.append(test_errors_value)
    
    theta_values.append(theta_opt_for_polynomial_features)
    
    

    probabilities_test = logistic_func(theta = theta_opt_for_polynomial_features, object_features = phi_2_dim_polynomial_features(df_test.drop('label', axis = 1).to_numpy(), polynomial_degree = polynomial_degree))
    y_true_test = df_test['label'].to_numpy()
    auc_value = plot_roc_curve(y_true = y_true_test, y_model = probabilities_test, data_set_name = "polynomial for k = " + str(polynomial_degree) )
    
    auc_values.append(auc_value)

plt.plot(polynomial_degree_values, train_errors_values, color = "green", label = "Train Data")
plt.plot(polynomial_degree_values, test_errors_values, color = "red", label = "Test Data")
best_k_value = polynomial_degree_values[np.argmin(test_errors_values)]

theta_opt_for_best_k = theta_values[np.argmin(test_errors_values)]

train_errors_value, test_errors_value = calculate_train_test_errors_for_given_model(df_train = df_train, df_test = df_test, features_design = phi_2_dim_polynomial_features, theta_optimal = theta_opt_for_best_k, model = logistic_func, polynomial_degree = best_k_value)


print("The best value of polynomial degree: ", polynomial_degree_values[np.argmin(test_errors_values)])
print("Number of errors on train data for the best k: ", train_errors_value)
print("Number of errors on test data for the best k: ", test_errors_value)
plt.title("Train and test errors as a function of polynomial features degree")
plt.xlabel('Polynomial degree')
plt.ylabel('Number of errors')
plt.legend()
plt.savefig('Train_and_test_errors_as_a_function_of_polynomial_features_degree.png')
plt.close()

# AUC values plot:

plt.plot(polynomial_degree_values, auc_values, label = "AUC value")
plt.title("AUC values as a function of polynomial features degree")
plt.xlabel('Polynomial degree')
plt.ylabel('AUC')
plt.legend()
plt.savefig('AUC_values_as_a_function_of_polynomial_features_degree.png')
plt.close()

get_final_results(polynomial_degree_values = polynomial_degree_values, train_errors_values = train_errors_values, test_errors_values = test_errors_values, auc_values = auc_values )