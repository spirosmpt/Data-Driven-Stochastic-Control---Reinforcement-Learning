% Parameters
n = 1000; % Number of states
s = linspace(-6, 15, n); % State grid
gamma = 0.8; % Discount factor
tolerance = 1e-200; % Convergence criterion

% Reward function
R = min(2, s.^2);

% Transition dynamics for action 1

mu1 = 0.8 * s + 1;
sigma1 = 1;
H1 = zeros(n, n);
for j = 1:n
  H1(:, j) = normcdf((s - mu1(j)) / sigma1);
end

% Compute F1 matrix
F1 = zeros(n, n);
for i = 1:n
  F1(i, 1) = (H1(2, i) - H1(1, i)) / 2;
  F1(i, 2) = (H1(3, i) - H1(1, i)) / 2;
  for j = 3:(n-1)

    F1(i, j) = (H1(j+1, i) - H1(j-1, i)) / 2;
  end
  F1(i, n) = (H1(n, i) - H1(n-1, i)) / 2;
end

% Transition dynamics for action 2
mu2 = -2;
sigma2 = 1;
H2 = zeros(n, n);
for j = 1:n
  H2(:, j) = normcdf((s - mu2) / sigma2);
end

% Compute F2 matrix
F2 = zeros(n, n);
for i = 1:n

  F2(i, 1) = (H2(2, i) - H2(1, i)) / 2;
  F2(i, 2) = (H2(3, i) - H2(1, i)) / 2;
  for j = 3:(n-1)
    F2(i, j) = (H2(j+1, i) - H2(j-1, i)) / 2;
  end
  F2(i, n) = (H2(n, i) - H2(n-1, i)) / 2;
end

% Initialize v1 and v2
v1 = zeros(n, 1);
v2 = zeros(n, 1);

% Iterate until convergence

while true
  v1_new = F1 * (R' + gamma * max(v1, v2));
  v2_new = F2 * (R' + gamma * max(v1, v2));

  % Check convergence
  if max(abs(v1_new - v1)) < tolerance && max(abs(v2_new - v2)) < tolerance
    break;
  end

  % Update v1 and v2
  v1 = v1_new;
  v2 = v2_new;
end

% Plot results
figure;

plot(s, v1, 'r', 'LineWidth', 2); hold on;
plot(s, v2, 'b', 'LineWidth', 2);
title('Value Functions v_1(S) and v_2(S)');
xlabel('State S');
ylabel('Value');
legend('v_1(S)', 'v_2(S)');
grid on;

save('v1_resultsnumerical_p3.mat', 's', 'v1', 'v2');
