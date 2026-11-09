% Βήμα 1: Δημιουργία δεδομένων
N = 500;           % Αριθμός δεδομένων
X = randn(N, 1);      % X ~ N(0, 1)
W = randn(N, 1);      % W ~ N(0, 1)
Y = 0.8 * X + W;     % Y = 0.8*X + W

% Βήμα 2: Αρχικοποίηση παραμέτρων
m = 50;            % Μέγεθος κρυφής στρώσης
W1 = randn(m, 1)*0.1;       % Τυχαία αρχικοποίηση βαρών 1ης στρώσης
b1 = randn(1, m)*0.1;       % Bias για την 1η στρώση
W2 = randn(m, 1)*0.1;        % Τυχαία αρχικοποίηση βαρών 2ης στρώσης
b2 = randn(1)*0.1;         % Bias εξόδου

% Adam Υπερπαράμετροι
mu = 0.0001;         % Learning rate
beta1 = 0.9;        % Συντελεστής για momentum
beta2 = 0.999;       % Συντελεστής για RMSprop
epsilon = 1e-310;      % Μικρή σταθερά για σταθερότητα
epochs = 5000;        % Αριθμός επαναλήψεων
costs = zeros(epochs, 1); % Αποθήκευση κόστους ανά εποχή

% Adam Αρχικοποίηση στιγμών
mW1 = zeros(size(W1)); vW1 = zeros(size(W1));
mb1 = zeros(size(b1)); vb1 = zeros(size(b1));
mW2 = zeros(size(W2)); vW2 = zeros(size(W2));
mb2 = 0; vb2 = 0;

% Βήμα 4: Εκπαίδευση με Adam Gradient Descent
for epoch = 1:epochs
 % Forward pass
 Z = X * W1' + b1;         % Υπολογισμός Z = X*W1' + b1
 H = max(0, Z);           % Ενεργοποίηση ReLU
 U = H * W2 + b2;           % Υπολογισμός εξόδου u(X)

 % Υπολογισμός κόστους
 phi_u = (exp(0.5 * abs(U)) - 1) + (1/3) * (exp(-1.5 * abs(U)) - 1); % φ(u)
 psi_u = 2 * sign(U) .* (exp(-0.5 * abs(U)) - 1);         % ψ(u)
 J = (1/N) * sum(phi_u + Y .* psi_u);                 % Κόστος J(u)
 costs(epoch) = J;                            % Αποθήκευση κόστους

 % Υπολογισμός παραγώγου
 dphi_u = 0.5 * sign(U) .* exp(0.5 * abs(U)) - 0.5 * sign(U) .* exp(-1.5 * abs(U));

dpsi_u = 2 * exp(-0.5 * abs(U)); % Ενημερώθηκε από τις εξισώσεις

% Υπολογισμός ω(u) και ρ(u)
omega_u = sinh(U);              % ω(u)
rho_u = -exp(-0.5 * abs(U));       % ρ(u)

% Παράγωγος του J ως προς U
dU = (1/N) * ((Y - omega_u) .* rho_u);

% Υπολογισμός gradients
dW2 = H' * dU;            % Gradient για W2
dB2 = sum(dU);             % Gradient για b2
dH = dU * W2';            % Gradient για H

dZ = dH .* (Z > 0);       % ReLU derivative
dW1 = X' * dZ;            % Gradient για W1
dB1 = sum(dZ, 1);              % Gradient για b1

% Adam ενημέρωση παραμέτρων
% 1η Στιγμή (momentum)
mW1 = beta1 * mW1 + (1 - beta1) * dW1';
mb1 = beta1 * mb1 + (1 - beta1) * dB1;
mW2 = beta1 * mW2 + (1 - beta1) * dW2;
mb2 = beta1 * mb2 + (1 - beta1) * dB2;

% 2η Στιγμή (RMSprop)
vW1 = beta2 * vW1 + (1 - beta2) * (dW1'.^2);
vb1 = beta2 * vb1 + (1 - beta2) * (dB1.^2);
vW2 = beta2 * vW2 + (1 - beta2) * (dW2.^2);
vb2 = beta2 * vb2 + (1 - beta2) * (dB2.^2);

 % Bias-corrected στιγμές
 mW1_hat = mW1 / (1 - beta1^epoch);
 mb1_hat = mb1 / (1 - beta1^epoch);
 mW2_hat = mW2 / (1 - beta1^epoch);
 mb2_hat = mb2 / (1 - beta1^epoch);

 vW1_hat = vW1 / (1 - beta2^epoch);
 vb1_hat = vb1 / (1 - beta2^epoch);
 vW2_hat = vW2 / (1 - beta2^epoch);
 vb2_hat = vb2 / (1 - beta2^epoch);

 % Ενημέρωση παραμέτρων

 W1 = W1 - mu * mW1_hat ./ (sqrt(vW1_hat) + epsilon);
 b1 = b1 - mu * mb1_hat ./ (sqrt(vb1_hat) + epsilon);
 W2 = W2 - mu * mW2_hat ./ (sqrt(vW2_hat) + epsilon);
 b2 = b2 - mu * mb2_hat ./ (sqrt(vb2_hat) + epsilon);

 % Εμφάνιση κόστους κάθε 100 εποχές
 if mod(epoch, 100) == 0
   fprintf('Epoch %d, Cost: %.4f\n', epoch, J);
 end
end

% Εξαγωγή γραφημάτων και αποτελεσμάτων (όπως πριν)

% Βήμα 5: Απεικόνιση αποτελεσμάτων
X_test = linspace(min(X), max(X), 1000)'; % Δημιουργία δεδομένων δοκιμής

Z_test = X_test * W1' + b1;             % Υπολογισμός Z για δοκιμαστικά δεδομένα
H_test = max(0, Z_test);                % Ενεργοποίηση ReLU
U_test = H_test * W2 + b2;               % Υπολογισμός εξόδου

load("results_p1_numeric.mat", "x_grid","U");
% Γραφική 1: Προσέγγιση E[Y|X]
figure;
scatter(X, Y, 15, 'blue', 'filled'); hold on;
plot(X_test, sinh(U_test), 'r', 'LineWidth', 2);
plot(x_grid, U, 'g', 'LineWidth', 2);
xlabel('X');
ylabel('E[Y|X]');

title('Data-Driven Estimate of E[Y|X]');
legend('Samples', 'Estimated E[Y|X]');

% Γραφική 2: Κόστος ανά Εποχή
figure;
plot(1:epochs, costs, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Cost');
title('Cost per Epoch');
grid on;

% Ground Truth Calculation
X_truth = linspace(min(X), max(X), 1000)'; % Τιμές του X για το ground truth
E_GY_given_X = zeros(size(X_truth));            % Αρχικοποίηση του ground truth

for i = 1:length(X_truth)
  % Σημείο στο X
  x = X_truth(i);

  % Παράμετροι της κανονικής Y|X
  mean_Y_given_X = 0.8 * x; % Μέση τιμή
  std_Y_given_X = 1;        % Τυπική απόκλιση

  % Ground Truth με Monte Carlo
 samples = mean_Y_given_X + std_Y_given_X * randn(1, 100000); % Δείγματα από N(0.8X, 1)
  % Υπολογισμός G(Y)
  E_GY_given_X(i) = mean(samples); % Μέση τιμή

end

% Γραφική 2: Κόστος ανά Εποχή
figure;
plot(1:epochs, costs, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Cost');
title('Cost per Epoch (Hypothesis C1)');
grid on;
