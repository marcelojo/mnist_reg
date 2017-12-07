function [prob pred] = test_regression (a, data)
    
    data = (data .- min(data)) ./ (max(data) .- min(data) + 1);
    
    % Sigmoid function     
    % h(x) = 1 / (1 + e^-z)
    % where z = a0x0 + a1x1 + ... + a4x4  ==> where x0 is always 1
    data_qty = size(data)(1);
    features_len = size(data)(2);
    
    % X0 is always one. Copy rest of the data
    x = ones(data_qty, features_len + 1);
    x(:,2:end) = data;
    
    % create vector and calculate all points
    z = x * a';            % y = a0x0 + a1x1 + a2x2 
    h = 1 ./ (1 + e.^-z);
    
    % Returns the probability to be in class 1 or 0
    [prob pred] = max(h, [], 2);
    
    pred = pred - 1;    % 0 is class zero not 10
    
endfunction