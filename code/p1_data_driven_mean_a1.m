% Βήμα 1: Δημιουργία δεδομένων
N = 500;           % Αριθμός δεδομένων
X = randn(N, 1);      % X ~ N(0, 1)
W = randn(N, 1);      % W ~ N(0, 1)
Y = 0.8 * X + W;     % Y = 0.8*X + W

% Βήμα 2: Αρχικοποίηση Νευρωνικού Δικτύου
m = 50;            % Μέγεθος κρυφής στρώσης

W1 = randn(m, 1);      % Τυχαία αρχικοποίηση βαρών 1ης στρώσης
b1 = randn(1, m);     % Bias για την 1η στρώση
W2 = randn(m, 1);      % Τυχαία αρχικοποίηση βαρών 2ης στρώσης
b2 = randn(1);       % Bias εξόδου

% Βήμα 3: Υπερπαράμετροι
mu = 0.001;          % Ρυθμός μάθησης
epochs = 1000;        % Αριθμός επαναλήψεων
costs = zeros(epochs, 1); % Αποθήκευση κόστους ανά εποχή

% Βήμα 4: Εκπαίδευση με Gradient Descent
for epoch = 1:epochs
 % Forward pass

 Z = X * W1' + b1;         % Υπολογισμός Z = X*W1' + b1
 H = max(0, Z);           % Ενεργοποίηση ReLU
 U = H * W2 + b2;          % Υπολογισμός εξόδου U(X)

 % Υπολογισμός κόστους
 J = (1/N) * sum(0.5 * U.^2 - Y .* U); % MSE
 costs(epoch) = J;            % Αποθήκευση κόστους

 % Υπολογισμός \omega(u) και \rho(u)
 omega_u = U;              % ω(u) = u
 rho_u = -1;            % ρ(u) = -1

 % Υπολογισμός gradient
 dU = (1/N) * (Y - omega_u) .* rho_u; % Gradient της εξόδου
 dW2 = H' * dU;              % Gradient για W2
 dB2 = sum(dU);               % Gradient για b2

  dH = dU * W2';                % Gradient για H
  dZ = dH .* (Z > 0);           % ReLU derivative
  dW1 = X' * dZ;               % Gradient για W1
  dB1 = sum(dZ, 1);               % Gradient για b1

  % Ενημέρωση παραμέτρων
  W2 = W2 - mu * dW2;
  b2 = b2 - mu * dB2;
  W1 = W1 - mu * dW1';
  b1 = b1 - mu * dB1;
end

% Βήμα 5: Απεικόνιση αποτελεσμάτων

X_test = linspace(min(X), max(X), 1000)'; % Δοκιμαστικά δεδομένα
Z_test = X_test * W1' + b1;             % Υπολογισμός Z για δοκιμαστικά δεδομένα
H_test = max(0, Z_test);                % Ενεργοποίηση ReLU
U_test = H_test * W2 + b2;               % Υπολογισμός εξόδου

load("results_p1_numeric.mat", "x_grid","U");
% Γραφική 1: Προσέγγιση E[Y|X]
figure;
scatter(X, Y, 15, 'blue', 'filled'); hold on; % Τα πραγματικά δεδομένα
plot(X_test, U_test, 'r', 'LineWidth', 2); % Η εκτίμηση του δικτύου
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
