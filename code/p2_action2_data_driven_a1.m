% Φόρτωση δεδομένων για Set 1

load('Set2_Pairs.mat'); % Δεδομένα (S_t, S_{t+1}, R(S_{t+1}))
X2 = set_2(:, 1); % Είσοδος (S_t) % Επόμενη κατάσταση S_{t+1}
Y2 = set_2(:, 2);
% Υπολογισμός της ανταμοιβής R(S_{t+1})
R1 = set_2(:, 3); % Ανταμοιβή για κάθε επόμενη κατάσταση

% Κανονικοποίηση δεδομένων

% Αρχικοποίηση παραμέτρων του νευρωνικού δικτύου
inputSize = 1; % Είσοδος (S_t)
hiddenSize = 100; % Κρυφό επίπεδο (m=100)
outputSize = 1; % Έξοδος

% Τυχαία αρχικοποίηση βαρών και biases (He initialization)
W1 = randn(hiddenSize, inputSize) * sqrt(2 / inputSize); % Βάρη εισόδου -> κρυφό επίπεδο

b1 = zeros(hiddenSize, 1);               % Biases για το κρυφό επίπεδο
W2 = randn(outputSize, hiddenSize) * sqrt(2 / hiddenSize); % Βάρη κρυφό -> έξοδος
b2 = 0;                         % Bias για την έξοδο

% Υπερπαράμετροι
learningRate = 0.001; % Ρυθμός μάθησης
epochs = 3000;      % Εποχές εκπαίδευσης

% Υπερπαράμετροι Adam
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-310;

% Αρχικοποίηση στιγμών για κάθε παράμετρο
[mW1, vW1] = deal(zeros(size(W1)));
[mW2, vW2] = deal(zeros(size(W2)));
[mb1, vb1] = deal(zeros(size(b1)));
[mb2, vb2] = deal(zeros(size(b2)));

% Επαναληπτική εκπαίδευση (Adam Optimizer)
costs = zeros(epochs, 1); % Αποθήκευση κόστους για κάθε εποχή
N = length(X2);
% Βήμα 4: Εκπαίδευση με Adam και νέο κόστος
for epoch = 1:epochs
 total_loss = 0;
 for i = 1:length(X2)
   % Είσοδος
   x = X2(i); % Τρέχον δείγμα
   r = R1(i); % Ανταμοιβή R(S_{t+1})

% FORWARD PASS
z1 = W1 * x + b1;      % Είσοδος στο κρυφό επίπεδο
a1 = max(0, z1);      % ReLU ενεργοποίηση
z2 = W2 * a1 + b2;      % Έξοδος
u = z2;         % Εκτίμηση του u(S_t, θ)

% Νέο κόστος
total_loss = total_loss + (0.5 * u^2 - r * u);

% Υπολογισμός \omega(u) και \rho(u)
omega_u = u;          % ω(u) = u
rho_u = -1;         % ρ(u) = -1

% BACKWARD PASS (Υπολογισμός gradients)
dU = (1 / N) * (r - omega_u) * rho_u; % Gradient της εξόδου
dW2 = a1' * dU;               % Gradient για W2
db2 = dU;                  % Gradient για b2
dW1 = dU * W2' .* (z1 > 0) * x;     % Gradient για W1
db1 = dU * W2' .* (z1 > 0);        % Gradient για b1

% Ενημέρωση παραμέτρων με Adam
% Update W1
mW1 = beta1 * mW1 + (1 - beta1) * dW1;
vW1 = beta2 * vW1 + (1 - beta2) * (dW1.^2);
mW1_hat = mW1 / (1 - beta1^epoch);
vW1_hat = vW1 / (1 - beta2^epoch);
W1 = W1 - learningRate * mW1_hat ./ (sqrt(vW1_hat) + epsilon);

   % Update b1
   mb1 = beta1 * mb1 + (1 - beta1) * db1;
   vb1 = beta2 * vb1 + (1 - beta2) * (db1.^2);
   mb1_hat = mb1 / (1 - beta1^epoch);
   vb1_hat = vb1 / (1 - beta2^epoch);
   b1 = b1 - learningRate * mb1_hat ./ (sqrt(vb1_hat) + epsilon);

   % Update W2
   mW2 = beta1 * mW2 + (1 - beta1) * dW2;
   vW2 = beta2 * vW2 + (1 - beta2) * (dW2.^2);
   mW2_hat = mW2 / (1 - beta1^epoch);
   vW2_hat = vW2 / (1 - beta2^epoch);
   W2 = W2 - learningRate * mW2_hat ./ (sqrt(vW2_hat) + epsilon);

   % Update b2
   mb2 = beta1 * mb2 + (1 - beta1) * db2;
   vb2 = beta2 * vb2 + (1 - beta2) * (db2.^2);
   mb2_hat = mb2 / (1 - beta1^epoch);
   vb2_hat = vb2 / (1 - beta2^epoch);
   b2 = b2 - learningRate * mb2_hat ./ (sqrt(vb2_hat) + epsilon);
 end

 % Μέσο κόστος για την εποχή
 costs(epoch) = total_loss / N;

 % Εκτύπωση κόστους κάθε 100 εποχές
 if mod(epoch, 100) == 0
   fprintf('Epoch %d/%d completed. Cost: %.6f\n', epoch, epochs,
costs(epoch));
 end

end

% Υπολογισμός του v1(X) (εκτιμήσεις του δικτύου)
v2 = zeros(size(X2));
for i = 1:length(X2)
  z1 = W1 * X2(i) + b1;
  a1 = max(0, z1); % ReLU ενεργοποίηση
  z2 = W2 * a1 + b2;
  v2(i) = z2; % Εκτίμηση
end
% Υπολογισμός του v1(X) (εκτιμήσεις του δικτύου)
x2_a1_test = linspace(min(X2),max(X2), 3000);

v2_a1_test = zeros(size(x2_a1_test));
for i = 1:length(x2_a1_test)
  z1 = W1 * x2_a1_test(i) + b1;
  a1 = max(0, z1); % ReLU ενεργοποίηση
  z2 = W2 * a1 + b2;
  v2_a1_test(i) = z2;
end

% Ανακανονικοποίηση του v1 για σύγκριση με τα δεδομένα

% Plot του κόστους
figure;
plot(1:epochs, costs, '-');
title('Cost per Epoch');
xlabel('Epoch');

ylabel('Cost (MSE)');

% Plot του v1(X) χωρίς γραμμή (μόνο σημεία)
figure;
plot(X2, v2, 'o'); % Χωρίς γραμμή, μόνο σημεία
title('v_2(X) vs X');
xlabel('X (S_t)');
ylabel('v_2(X)');

% Αποθήκευση των X1 και v1 σε ένα αρχείο .mat
save('v2_results.mat', 'x2_a1_test', 'v2_a1_test');
