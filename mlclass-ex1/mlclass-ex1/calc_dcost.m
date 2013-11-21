function [d_cost] = calc_dcost(X, y, theta, j)
    d_cost = 0;
    m = length(y);
    for i = 1:m
        d_cost = d_cost + (X(i,:) * theta - y(i)) * X(i,j);
    end

    d_cost = (1/m) * d_cost;
end