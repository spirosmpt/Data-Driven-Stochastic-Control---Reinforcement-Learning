% Define grids
x_grid = linspace(-5, 5, 500);
y_grid = linspace(-10, 10, 500);
G = y_grid;

mu = 0;
sigma = 1;
a = @(x) 0.8 * x;

H = zeros(length(y_grid), length(x_grid));
for j = 1:length(x_grid)
    H(:, j) = normcdf((y_grid - a(x_grid(j)) - mu) / sigma);
end

n = length(y_grid) - 1;
m = length(x_grid) - 1;
F = zeros(m+1, n+1);
for i = 1:(m+1)
    F(i, 1) = (H(2, i) - H(1, i)) / 2;
    F(i, 2) = (H(3, i) - H(1, i)) / 2;
    for j = 3:n
        F(i, j) = (H(j+1, i) - H(j-1, i)) / 2;
    end
    F(i, n) = (H(n+1, i) - H(n-1, i)) / 2;
    F(i, n+1) = (H(n+1, i) - H(n, i)) / 2;
end

disp('Size of F matrix:');
disp(size(F));
disp('Sample rows of F matrix:');
disp(F(1:5, 1:5));

U = F * G';
figure;
plot(x_grid, U, 'LineWidth', 2);
xlabel('X');
ylabel('E[Y | X]');
title('Conditional Expectation E[Y | X]');
grid on;
p = polyfit(x_grid, U, 1);
slope = p(1);
disp(['The slope of the graph is: ', num2str(slope)]);
save("results_p1_numeric.mat", "x_grid","U");