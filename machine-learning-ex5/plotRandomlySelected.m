function plotRandomlySelected(X, y, Xval, yval, lambda, loops)
    % For small datasets it is often helpful - when plotting learning 
    % curves to debug - to average across multiple sets of RANDOMLY 
    % selected examples => training error, cross validation error
    
    % 1) randomly select i examples from the training set and
    %    i examples from the cross validation set.
    % 2) learn theta using randomly selected set
    % 3) evaluate the parameters on the randomly chosen training set and 
    %    cross validation set.
    % 4) repeat multiple times (say loops = 50)
    % 5) average error => training error, cv error for i examples
   
    m = size(X, 1);
    
    error_train = zeros(m, 1);
    error_val   = zeros(m, 1);
    
    
    for i = 1:m
        error_train_sum = 0;  
        error_val_sum = 0;  
        
        % i = number of training examples
        for l = 1:loops
            % randomly select i rows, pick first ith rows
            randRows = randperm(m);
            randRows = randRows(randRows(1:i));
            
            X_rand = X(randRows, :);
            y_rand = y(randRows);
            
            X_rand_val = Xval(randRows, :);
            y_rand_val = yval(randRows);
            
            % Learn theta/parameters on X_rand, y_rand
            theta = trainLinearReg(X_rand, y_rand, lambda);
            
            % Evaluate theta on training set and cv set
            [J_train, grad_train] = linearRegCostFunction(X_rand, y_rand, theta, 0);
            [J_val, grad_val] = linearRegCostFunction(X_rand_val, y_rand_val, theta, 0);
            
            % Accumalate error
            error_train_sum = error_train_sum + J_train;
            error_val_sum = error_val_sum + J_val;
        end
        
        error_train(i) = error_train_sum / loops;
        error_val(i) = error_val_sum / loops;
    end
    
   
    % plot
    figure();
    title(sprintf('Learning Curve w/ randomly selected examples (lambda = %f, loops = %f)', lambda, loops));
    plot(1:m, error_train, 1:m, error_val);
    xlabel('Number of training examples');
	ylabel('Error');
    axis([0 13 0 100]);
	legend('Train', 'Cross Validation');

end