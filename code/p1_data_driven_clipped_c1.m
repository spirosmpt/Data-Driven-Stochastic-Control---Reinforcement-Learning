% Βήμα 1: Δημιουργία δεδομένων
N = 500;               % Αριθμός δεδομένων
X = randn(N, 1);           % X ~ N(0, 1)
W = randn(N, 1);            % W ~ N(0, 1)
Y = 0.8 * X + W;          % Y = 0.8*X + W

% Υπολογισμός G(Y) (στο διάστημα [-1, 1])
G = min(1, max(-1, Y));        % G(Y) = min{1, max{-1, Y}}

% Βήμα 2: Αρχικοποίηση παραμέτρων \( \theta \)
m = 50;                % Μέγεθος κρυφής στρώσης

% Δημιουργία δομής θ (theta) για αποθήκευση όλων των παραμέτρων του δικτύου
theta = struct();
theta.W1 = randn(m, 1) * 0.1; % Βάρη για την 1η στρώση
theta.b1 = randn(1, m) * 0.1; % Bias για την 1η στρώση

theta.W2 = randn(m, 1) * 0.1; % Βάρη για την 2η στρώση
theta.b2 = randn(1) * 0.1; % Bias εξόδου

% Αρχικοποίηση Adam
alpha = 0.0001;          % Ρυθμός μάθησης
beta1 = 0.9;          % Παράμετρος για κινητό μέσο των gradients
beta2 = 0.999;          % Παράμετρος για κινητό μέσο των τετραγώνων των
gradients
epsilon = 1e-8;         % Σταθερά για αποφυγή διαίρεσης με το μηδέν

% Adam momentum και velocity terms
mW1 = zeros(size(theta.W1)); vW1 = zeros(size(theta.W1));
mb1 = zeros(size(theta.b1)); vb1 = zeros(size(theta.b1));

mW2 = zeros(size(theta.W2)); vW2 = zeros(size(theta.W2));
mb2 = 0; vb2 = 0;

% Υπερπαράμετροι
epochs = 5000;           % Αριθμός επαναλήψεων
costs = zeros(epochs, 1);     % Αποθήκευση κόστους ανά εποχή

% Συναρτήσεις για την Υπόθεση C1
omega = @(z) -1 ./ (1 + exp(z)) + (exp(z) ./ (1 + exp(z))); % \omega(z)
rho = @(z) -exp(z) ./ (1 + exp(z));               % \rho(z)
phi = @(z) (2 ./ (1 + exp(z))) + log1p(exp(z));        % \phi(z)
psi = @(z) -log1p(exp(z));                   % \psi(z)

% Βήμα 3: Εκπαίδευση με Adam Optimizer
for epoch = 1:epochs
  % Forward pass
  Z1 = X * theta.W1' + theta.b1;      % Υπολογισμός Z1 = X*W1' + b1

 H1 = max(0, Z1);             % Ενεργοποίηση ReLU
 U = H1 * theta.W2 + theta.b2;      % Υπολογισμός εξόδου U(X)

 % Υπολογισμός κόστους
 J = (1/N) * sum(phi(U) + G .* psi(U)); % Χρήση \phi και \psi από C1
 costs(epoch) = J;            % Αποθήκευση κόστους
 lambda = 0.01; % Συντελεστής Regularization
 %J = J + (lambda / (2 * N)) * (sum(theta.W1(:).^2) + sum(theta.W2(:).^2));

 % Υπολογισμός gradient
 omega_u = omega(U);               % \omega(U)
 rho_u = rho(U);             % \rho(U)

 dU = (1/N) * (G - omega_u) .* rho_u; % Gradient της εξόδου
 dW2 = H1' * dU;              % Gradient για W2
 dB2 = sum(dU);               % Gradient για b2
 dH1 = dU * theta.W2';           % Gradient για H1
 dZ1 = dH1 .* (Z1 > 0);         % Παράγωγος της ReLU
 dW1 = X' * dZ1;             % Gradient για W1
 dB1 = sum(dZ1, 1);             % Gradient για b1

 % Adam updates
 mW2 = beta1 * mW2 + (1 - beta1) * dW2;
 vW2 = beta2 * vW2 + (1 - beta2) * (dW2 .^ 2);
 mb2 = beta1 * mb2 + (1 - beta1) * dB2;
 vb2 = beta2 * vb2 + (1 - beta2) * (dB2 .^ 2);
 theta.W2 = theta.W2 - alpha * (mW2 / (1 - beta1^epoch)) ./ (sqrt(vW2 / (1 -
beta2^epoch)) + epsilon);
 theta.b2 = theta.b2 - alpha * (mb2 / (1 - beta1^epoch)) ./ (sqrt(vb2 / (1 -
beta2^epoch)) + epsilon);

  mW1 = beta1 * mW1 + (1 - beta1) * dW1';
  vW1 = beta2 * vW1 + (1 - beta2) * (dW1' .^ 2);
  mb1 = beta1 * mb1 + (1 - beta1) * dB1;
  vb1 = beta2 * vb1 + (1 - beta2) * (dB1 .^ 2);
 theta.W1 = theta.W1 - alpha * (mW1 / (1 - beta1^epoch)) ./ (sqrt(vW1 / (1 -
beta2^epoch)) + epsilon);
 theta.b1 = theta.b1 - alpha * (mb1 / (1 - beta1^epoch)) ./ (sqrt(vb1 / (1 -
beta2^epoch)) + epsilon);
end

% Βήμα 4: Απεικόνιση αποτελεσμάτων
X_test = linspace(-6, 6, 3000)';    % Δοκιμαστικά δεδομένα

Z1_test = X_test * theta.W1' + theta.b1; % Υπολογισμός Z1 για δοκιμαστικά δεδομένα
H1_test = max(0, Z1_test);         % Ενεργοποίηση ReLU
U_test = H1_test * theta.W2 + theta.b2; % Υπολογισμός εξόδου

% Απεικόνιση Ground Truth
load("results_p1_numeric_clipped.mat", "x_grid","U");
figure;
scatter(X, G, 15, 'blue', 'filled'); hold on; % Τα πραγματικά δεδομένα
plot(X_test, omega(U_test), 'r', 'LineWidth', 2);
plot(x_grid, U, 'g', 'LineWidth', 2);% Η εκτίμηση του δικτύου (μετά από ω(U))
xlabel('X');
ylabel('E[G(Y)|X]');
title('Data-Driven Estimate of E[G(Y)|X] (Hypothesis C1)');
legend('Samples', 'Estimated E[G(Y)|X]');
grid on;

% Γραφική 2: Κόστος ανά Εποχή
figure;
plot(1:epochs, costs, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Cost');
title('Cost per Epoch (Hypothesis C1)');
grid on;
