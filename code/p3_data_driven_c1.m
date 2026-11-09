% Φόρτωση δεδομένων για Set 1

load('Set1_Pairs.mat'); % Δεδομένα για α = 1
load('Set2_Pairs.mat');
X1 = set_1(:, 1); % Είσοδος (S_t)
Y1 = set_1(:, 2); % Επόμενη κατάσταση (S_{t+1})
R1 = set_1(:, 3); % Ανταμοιβή R(S_{t+1})
X2 = set_2(:, 1); % Είσοδος (S_t)
Y2 = set_2(:, 2); % Επόμενη κατάσταση (S_{t+1})
R2 = set_2(:, 3); % Ανταμοιβή R(S_{t+1})

a=0;
b=10;
omega =@(z) a ./ (1 + exp(z)) + b .* (exp(z) ./ (1 + exp(z)));
rho =@(z) -exp(z) ./ (1 + exp(z));
phi =@(z) ((b - a) ./ (1 + exp(z))) + (b .* log(1 + exp(z)));
psi =@(z) -log(1 + exp(z));
% Υπερπαράμετροι

learningRate = 0.001; % Ρυθμός μάθησης (mu)
gamma = 0.8;      % Discount factor
epochs = 1000;     % Αριθμός εποχών εκπαίδευσης
m = 100; % Μέγεθος του κρυφού επιπέδου
inputSize = 1;    % Μέγεθος εισόδου
outputSize = 1;   % Μέγεθος εξόδου

% Αρχικοποίηση παραμέτρων του νευρωνικού δικτύου (He initialization)
W1_1 = randn(m, inputSize) * sqrt(2 / inputSize)*0.1; % Βάρη εισόδου -> κρυφό επίπεδο
b1_1= zeros(m, 1)*0.1;                % Biases για το κρυφό επίπεδο
W2_1= randn(outputSize, m) * sqrt(2 / m)*0.1; % Βάρη κρυφό -> έξοδος

b2_1= 0;                       % Bias για την έξοδο

W1_2 = randn(m, inputSize) * sqrt(2 / inputSize)*0.1; % Βάρη εισόδου -> κρυφό επίπεδο
b1_2 = zeros(m, 1)*0.1;               % Biases για το κρυφό επίπεδο
W2_2 = randn(outputSize, m) * sqrt(2 / m)*0.1; % Βάρη κρυφό -> έξοδος
b2_2 = 0;                      % Bias για την έξοδο

% Αποθήκευση κόστους για κάθε εποχή
costs_1= zeros(epochs, 1);
costs_2= zeros(epochs, 1);
total_loss_1 = 0; % Αρχικοποίηση συνολικής απώλειας για το πρώτο νευρωνικό
total_loss_2 = 0; % Αρχικοποίηση συνολικής απώλειας για το δεύτερο νευρωνικό
% Επαναληπτική εκπαίδευση
for epoch = 1:epochs
 total_loss_1 = 0; % Συνολική απώλεια για το πρώτο forward pass

total_loss_2 = 0; % Συνολική απώλεια για το δεύτερο forward pass

for i = 1:min(size(X1, 1), size(X2, 1))
  % Είσοδος και στόχος για το δείγμα
  x1 = X1(i);
  r1 = R1(i);
  x2 = X2(i);
  r2 = R2(i);
  y1 = Y1(i);
  y2 = Y2(i);

  % --- Πρώτο Forward Pass (x1, y1, y2) ---
  % Υπολογισμός για x1

  z1_x1 = W1_1 * x1 + b1_1;
  a1_x1 = max(0, z1_x1); % ReLU ενεργοποίηση
  z2_x1 = W2_1 * a1_x1 + b2_1;
  u_x1 = z2_x1;

  % Υπολογισμός για y1
  z1_y1 = W1_1 * y1 + b1_1;
  a1_y1 = max(0, z1_y1); % ReLU ενεργοποίηση
  z2_y1 = W2_1 * a1_y1 + b2_1;
  u_y1 = z2_y1;
   % Υπολογισμός για y1_u2
  z1_y1_u2 = W1_2 * y1 + b1_2;
  a1_y1_u2 = max(0, z1_y1_u2); % ReLU ενεργοποίηση
  z2_y1_u2 = W2_2 * a1_y1_u2 + b2_2;
  u_y1_u2= z2_y1_u2;
  % Υπολογισμός για x2

   z1_x2 = W1_2 * x2 + b1_2;
   a1_x2 = max(0, z1_x2); % ReLU ενεργοποίηση
   z2_x2 = W2_2 * a1_x2 + b2_2;
   u_x2 = z2_x2;
   % Υπολογισμός για y2
   z1_y2 = W1_2 * y2 + b1_2;
   a1_y2 = max(0, z1_y2); % ReLU ενεργοποίηση
   z2_y2 = W2_2 * a1_y2 + b2_2;
   u_y2 = z2_y2;
       % Υπολογισμός για y2_u1
   z1_y2_u1 = W1_1 * y2 + b1_1;
   a1_y2_u1 = max(0, z1_y2_u1); % ReLU ενεργοποίηση
   z2_y2_u1 = W2_1 * a1_y2_u1 + b2_1;

   u_y2_u1 = z2_y2_u1;
% --- Δεύτερο Forward Pass (x2, y1, y2) ---

   % Υπολογισμός κόστους για το πρώτο νευρωνικό
   omega_u_x1 = omega(u_x1);
   omega_u_y1 = omega(u_y1);
   omega_u_y2 = omega(u_y2);
   omega_u_x2 = omega(u_x2);
   omega_u_y1_u2 = omega(u_y1_u2);
   omega_u_y2_u1 = omega(u_y2_u1);

   rho_x1 = rho(u_x1);
   rho_x2 = rho(u_x2);
  cost_1 = ((r1 + gamma * max(omega_u_y1, omega_u_y1_u2) -
omega_u_x1)*(rho_x1));
   total_loss_1 = total_loss_1 + cost_1;

   % Υπολογισμός κόστους για το δεύτερο νευρωνικό
  cost_2 = (r2 + gamma * max(omega_u_y2_u1, omega_u_y2) -
omega_u_x2)*(rho_x2);
   total_loss_2 = total_loss_2 + cost_2;

   % --- Backpropagation για το πρώτο νευρωνικό ---
   grad_W2_1 = cost_1 * a1_x1'; % Gradient για τα βάρη του δεύτερου επιπέδου
   grad_b2_1 = cost_1; % Gradient για το bias του δεύτερου επιπέδου
   delta_a1_x1 = W2_1' * cost_1;
   delta_z1_x1 = delta_a1_x1 .* (z1_x1 > 0); % ReLU παράγωγος
   grad_W1_1 = delta_z1_x1 * x1'; % Gradient για τα βάρη του πρώτου επιπέδου

   grad_b1_1 = delta_z1_x1; % Gradient για το bias του πρώτου επιπέδου

   % Ενημέρωση παραμέτρων για το πρώτο νευρωνικό
   W2_1 = W2_1 - learningRate * grad_W2_1;
   b2_1 = b2_1 - learningRate * grad_b2_1;
   W1_1 = W1_1 - learningRate * grad_W1_1;
   b1_1 = b1_1 - learningRate * grad_b1_1;

   % --- Backpropagation για το δεύτερο νευρωνικό ---
   grad_W2_2 = cost_2 * a1_x2'; % Gradient για τα βάρη του δεύτερου επιπέδου
   grad_b2_2 = cost_2; % Gradient για το bias του δεύτερου επιπέδου
   delta_a1_x2 = W2_2' * cost_2;
   delta_z1_x2 = delta_a1_x2 .* (z1_x2 > 0); % ReLU παράγωγος
   grad_W1_2 = delta_z1_x2 * x2'; % Gradient για τα βάρη του πρώτου επιπέδου
   grad_b1_2 = delta_z1_x2; % Gradient για το bias του πρώτου επιπέδου

   % Ενημέρωση παραμέτρων για το δεύτερο νευρωνικό

    W2_2 = W2_2 - learningRate * grad_W2_2;
    b2_2 = b2_2 - learningRate * grad_b2_2;
    W1_2 = W1_2 - learningRate * grad_W1_2;
    b1_2 = b1_2 - learningRate * grad_b1_2;
  end

  % Αποθήκευση κόστους ανά εποχή
  costs_1(epoch) = total_loss_1 / min(size(X1, 1), size(X2, 1));
  costs_2(epoch) = total_loss_2 / min(size(X1, 1), size(X2, 1));
end

% Αποθήκευση εξόδων για όλες τις τιμές του x1 και x2
outputs_u_x1 = zeros(size(X1));

outputs_u_x2 = zeros(size(X2));

% Υπολογισμός εξόδων για κάθε τιμή x1 και x2
for c = 1:length(X1)
  % Είσοδος και στόχος για το δείγμα
  x1 = X1(c);
  % --- Forward Pass για x1 ---
  z1_x1 = W1_1 * x1 + b1_1;
  a1_x1 = max(0, z1_x1); % ReLU ενεργοποίηση
  z2_x1 = W2_1 * a1_x1 + b2_1;
  outputs_u_x1(c) = omega(z2_x1); % Αποθήκευση της εξόδου u(x1)

end
for f = 1:length(X2)

  x2 = X2(f);

  z1_x2 = W1_2 * x2 + b1_2;
  a1_x2 = max(0, z1_x2); % ReLU ενεργοποίηση
  z2_x2 = W2_2 * a1_x2 + b2_2;
  outputs_u_x2(f) = omega(z2_x2);

end
% Ταξινόμηση δεδομένων (αν χρειάζεται)
[X1_sorted, sort_idx] = sort(X1);
outputs_u_x1_sorted = outputs_u_x1(sort_idx);
[X2_sorted, sort_idx] = sort(X2);
outputs_u_x2_sorted = outputs_u_x2(sort_idx);

load('v1_resultsnumerical_p3.mat', 's', 'v1', 'v2'); % Αποτελέσματα από τον πρώτο κώδικα

% Plot εξόδων του νευρωνικού με μία γραμμή
figure;
plot(s, v1, 'r-', 'LineWidth', 2, 'DisplayName', 'v1 Numerical '); hold on;
plot(s, v2, 'y-', 'LineWidth', 2, 'DisplayName', 'v2 Numerical ');
plot(X1_sorted, outputs_u_x1_sorted, 'k-', 'LineWidth', 2, 'MarkerSize', 1, 'DisplayName', 'v1 Data Driven (C1)');

plot(X2_sorted, outputs_u_x2_sorted, '-', 'LineWidth', 2, 'MarkerSize', 1, 'DisplayName', 'v2 Data Driven (C1)');
xlabel('S');
ylabel('Output');
title('Problem 3 Numeric - Data Driven for C1');
legend('show');
grid on;

figure;
plot(1:epochs, costs_1, '-', 'LineWidth', 2, 'MarkerSize', 1, 'DisplayName', 'Action 1
(C1)');
hold on;
plot(1:epochs, costs_2, '-', 'LineWidth', 2, 'MarkerSize', 1, 'DisplayName', 'Action 2
(C1)');
xlabel('Epochs');
ylabel('Cost');
title('Problem 3 Cost/Epoch for C1');
ylim([-0.06,0.12]);
legend('show');

grid on;
