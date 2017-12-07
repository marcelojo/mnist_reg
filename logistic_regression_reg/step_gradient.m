function [new_a J] = step_gradient(a, data, label, learning_rate, lambda)
    % Get data length
    data_len = size(data)(1);
    feat_len = size(data)(2);
    
    % Get x and y points from data
    x = ones(data_len, feat_len + 1); %x0 = 1
    x(:,2:end) = data;
    
    y = label;
    
    % For regularization a0 actually must be 0 (zero) otherwise we´ll kill the bias value
    a0 = a;
    if lambda != 0
      a0(1) = 0;
    end
    
    % gradiente is defined as:
    % (1/n) * sum(i=0:n){x_m*[h - y]}  where x0 is ALWAYS 1  
    
    % h(x) = 1 / 1 + e^-z (logistic function)
    z = x * a';                 % [60000 x 785] x [785 x 1] ==> y = a0x0 + a1x1 + a2x2 + ... a4x4
    h = 1 ./ (1 + e.^-z);       % [60000 x 1]
    
    % [1 x 60000] x [60000 x 785] ==> sum(i=0:n){x_m*[y' - y]} - a*lambda/m where x_m can be x0, x1, x2, ... , x4
    % Regularized Logistic Regression
    new_a = a - learning_rate * (1/data_len) * ( (h - y)' * x - lambda * a0);
    
    J = 1/data_len * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + lambda / (2 * data_len) * a0 * a0';

    %{
    x = [60k x 5]     - 60k samples
    a = [1 x 785]     - 785 parameters
    z = [60k x 1]     - 60k results
    h = [60k x 1]     - 60k results 
    new_a = [1 x 785] - 785 new parameters
    %}
endfunction
