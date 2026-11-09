% Ορισμός πλέγματος καταστάσεων
n = 1000; % Αριθμός σημείων
s = linspace(-6,20, n); % Πλέγμα καταστάσεων (S_t και S_{t+1})

% Συνάρτηση ανταμοιβής R(S) = min(2, S^2)
R = min(2, s.^2);

% Δυναμικές για α = 1
mu1 = 0.8 * s + 1.0; % Μέση τιμή των δυναμικών
sigma1 = 1;         % Τυπική απόκλιση

% Υπολογισμός H(y|x) για α = 1
H1 = zeros(n, n);
for j = 1:n
  H1(:, j) = normcdf((s - mu1(j)) / sigma1);

end

% Υπολογισμός F1
F1 = zeros(n, n);
for i = 1:n
  F1(i, 1) = (H1(2, i) - H1(1, i)) / 2;
  for j = 2:(n-1)
    F1(i, j) = (H1(j+1, i) - H1(j-1, i)) / 2;
  end
  F1(i, n) = (H1(n, i) - H1(n-1, i)) / 2;
end

% Υπολογισμός v1(X)

v1_num = F1 * R';

% Δυναμικές για α = 2
mu2 = -2.0; % Μέση τιμή των δυναμικών
sigma2 = 1; % Τυπική απόκλιση

% Υπολογισμός H(y|x) για α = 2
H2 = zeros(n, n);
for j = 1:n
  H2(:, j) = normcdf((s - mu2) / sigma2);
end

% Υπολογισμός F2
F2 = zeros(n, n);
for i = 1:n
  F2(i, 1) = (H2(2, i) - H2(1, i)) / 2;

  for j = 2:(n-1)
    F2(i, j) = (H2(j+1, i) - H2(j-1, i)) / 2;
  end
  F2(i, n) = (H2(n, i) - H2(n-1, i)) / 2;
end

% Υπολογισμός v2(X)
v2_num = F2 * R';

% Βέλτιστη πολιτική
optimal_policy = zeros(1, n);
for i = 1:n
  if v1_num(i) >= v2_num(i)

    optimal_policy(i) = 1;
  else
    optimal_policy(i) = 2;
  end
end

% Γραφήματα
figure;
plot(s, v1_num, 'r', 'LineWidth', 2); hold on;
plot(s, v2_num, 'b', 'LineWidth', 2);
title('Value Functions v_1(X) and v_2(X)');
xlabel('X');
ylabel('v(X)');
legend('v_1(X)', 'v_2(X)');
grid on;

figure;
plot(s, R, 'g', 'LineWidth', 2);
title('Reward Function R(S) = min{2, S^2}');
xlabel('S');
ylabel('R(S)');
grid on;

% Αποθήκευση
save('v1_resultsnumerical.mat', 's', 'v1_num', 'v2_num', 'optimal_policy');
save('v2_resultsnumerical.mat', 's', 'v2_num', 'v2_num', 'optimal_policy');
