% Define grids with restricted X range
x_grid = linspace(-6, 6, 500); % Restricted X grid (m+1 points)
y_grid = linspace(-50, 50, 500); % Y grid (n+1 points)

% Define G(Y) = min(1, max(-1, Y))
G = min(1, max(-1, y_grid)); % Apply transformation to y_grid

% Parameters for W ~ N(mu, sigma^2)
mu = 0;      % Mean of W
sigma = 1;     % Standard deviation of W
a = @(x) 0.8 * x; % a(X)

% Define conditional CDF H(y | X)
H = zeros(length(y_grid), length(x_grid)); % Initialize H matrix
for j = 1:length(x_grid)
  % Use the theory for conditional CDF

  H(:, j) = normcdf((y_grid - a(x_grid(j)) - mu) / sigma);
end

% Compute F matrix
n = length(y_grid) - 1; % Number of intervals in y_grid (n+1 points)
m = length(x_grid) - 1; % Number of intervals in x_grid (m+1 points)
F = zeros(m+1, n+1); % Initialize F matrix

for i = 1:(m+1)
  % First term (difference of 1 sample)
  F(i, 1) = (H(2, i) - H(1, i)) / 2;

  % Second term (difference of 2 samples)

  F(i, 2) = (H(3, i) - H(1, i)) / 2;

  % General terms (difference of 2 samples)
  for j = 3:n
    F(i, j) = (H(j+1, i) - H(j-1, i)) / 2;
  end

  % Penultimate term (difference of 2 samples)
  F(i, n) = (H(n+1, i) - H(n-1, i)) / 2;

  % Last term (difference of 1 sample)
  F(i, n+1) = (H(n+1, i) - H(n, i)) / 2;
end

% Compute U(x) as F * G
U = F * G';

% Plot the results
figure;
plot(x_grid, U, 'LineWidth', 2);
xlabel('X');
ylabel('E[\min{1, max{-1, Y}} | X]');
title('Conditional Expectation E[\min{1, max{-1, Y}} | X] (Restricted X Range)');
grid on;

% Compute the slope of the line using polyfit
p = polyfit(x_grid, U, 1); % Fit a line to the data
slope = p(1); % The slope is the first coefficient

% Display the slope
disp(['The slope of the graph is: ', num2str(slope)]);

save("results_p1_numeric_clipped.mat", "x_grid","U");
