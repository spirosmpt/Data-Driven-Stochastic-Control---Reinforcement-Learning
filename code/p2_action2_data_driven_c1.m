load('Set2_Pairs.mat'); % Δεδομένα (S_t, S_{t+1}, R(S_{t+1}))
X2_c1 = set_2(:, 1); % Είσοδος (S_t)
R2_c1= set_2(:, 3); % Ανταμοιβή R(S_{t+1})

% Κανονικοποίηση δεδομένων

% Αρχικοποίηση παραμέτρων του νευρωνικού δικτύου
inputSize = 1; % Είσοδος (S_t)
hiddenSize = 100; % Κρυφό επίπεδο (m=100)

outputSize = 1; % Έξοδος

% Τυχαία αρχικοποίηση βαρών και biases (He initialization)
W1 = randn(hiddenSize, inputSize) * sqrt(2 / inputSize); % Βάρη εισόδου -> κρυφό επίπεδο
b1 = zeros(hiddenSize, 1)*0.1;                     % Biases για το κρυφό επίπεδο
W2 = randn(outputSize, hiddenSize) * sqrt(2 / hiddenSize); % Βάρη κρυφό -> έξοδος
b2 = 0;                            % Bias για την έξοδο

% Υπερπαράμετροι για το ADAM
learningRate = 0.01; % Ρυθμός μάθησης
beta1 = 0.9;       % Υπερπαράμετρος για τη στιγμιαία μέση τιμή

beta2 = 0.999;       % Υπερπαράμετρος για τη στιγμιαία διακύμανση
epsilon = 1e-8;      % Μικρό σταθερό για αποφυγή διαίρεσης με το 0

% Αρχικοποίηση στιγμών για κάθε παράμετρο
[mW1, vW1] = deal(zeros(size(W1)));
[mW2, vW2] = deal(zeros(size(W2)));
[mb1, vb1] = deal(zeros(size(b1)));
[mb2, vb2] = deal(zeros(size(b2)));

a = 0;
b = 2;

% Υπολογισμός τιμών
omega =@(z) a ./ (1 + exp(z)) + b .* (exp(z) ./ (1 + exp(z)));
rho =@(z) -exp(z) ./ (1 + exp(z));
phi =@(z) ((b - a) ./ (1 + exp(z))) + (b .* log(1 + exp(z)));

psi =@(z) -log(1 + exp(z));

% Επαναληπτική εκπαίδευση (με ADAM)
epochs = 2000; % Αριθμός εποχών
costs = zeros(epochs, 1); % Αποθήκευση κόστους για κάθε εποχή

N = length(X2_c1);
for epoch = 1:epochs
  total_loss = 0;
  for i = 1:length(X2_c1)
    % Είσοδος
    x = X2_c1(i); % Τρέχον δείγμα
    r = R2_c1(i); % Ανταμοιβή R(S_{t+1})

    % FORWARD PASS
    z1 = W1 * x + b1;
    a1 = max(0, z1); % ReLU ενεργοποίηση
    z2 = W2 * a1 + b2;
    u = z2;

    % Υπολογισμός του ω(z) και του ρ(z)
    omega_u = omega(u);
    rho_u = rho(u);
        % Εξασφάλιση ότι οι τιμές παραμένουν στο [a, b]
    u = max(a, min(b, u));
    % Υπολογισμός κόστους
    total_loss = total_loss + phi(u) + r .* psi(u);

    % BACKWARD PASS (Υπολογισμός gradients)

% BACKWARD PASS (Υπολογισμός gradients)
dU = (r - omega_u) * rho_u; % Gradient της εξόδου
dW2 = a1' * dU;               % Gradient για W2
db2 = dU;                % Gradient για b2
dW1 = dU * W2' .* (z1 > 0) * x;    % Gradient για W1
db1 = dU * W2' .* (z1 > 0);       % Gradient για b1

% ADAM Updates για W1
mW1 = beta1 * mW1 + (1 - beta1) * dW1;
vW1 = beta2 * vW1 + (1 - beta2) * (dW1.^2);
mW1_hat = mW1 / (1 - beta1^epoch);
vW1_hat = vW1 / (1 - beta2^epoch);
W1 = W1 - learningRate * mW1_hat ./ (sqrt(vW1_hat) + epsilon);

% ADAM Updates για b1
mb1 = beta1 * mb1 + (1 - beta1) * db1;
vb1 = beta2 * vb1 + (1 - beta2) * (db1.^2);
mb1_hat = mb1 / (1 - beta1^epoch);
vb1_hat = vb1 / (1 - beta2^epoch);
b1 = b1 - learningRate * mb1_hat ./ (sqrt(vb1_hat) + epsilon);

% ADAM Updates για W2
mW2 = beta1 * mW2 + (1 - beta1) * dW2;
vW2 = beta2 * vW2 + (1 - beta2) * (dW2.^2);
mW2_hat = mW2 / (1 - beta1^epoch);
vW2_hat = vW2 / (1 - beta2^epoch);
W2 = W2 - learningRate * mW2_hat ./ (sqrt(vW2_hat) + epsilon);

% ADAM Updates για b2

    mb2 = beta1 * mb2 + (1 - beta1) * db2;
    vb2 = beta2 * vb2 + (1 - beta2) * (db2.^2);
    mb2_hat = mb2 / (1 - beta1^epoch);
    vb2_hat = vb2 / (1 - beta2^epoch);
    b2 = b2 - learningRate * mb2_hat ./ (sqrt(vb2_hat) + epsilon);
 end

 % Υπολογισμός μέσου κόστους για την εποχή
 costs(epoch) = total_loss / length(X2_c1);

 % Εκτύπωση προόδου κάθε 100 εποχές
 if mod(epoch, 100) == 0
   fprintf('Epoch %d/%d completed. Cost: %.6f\n', epoch, epochs,
costs(epoch));
 end
end

% Υπολογισμός του v1(X) (εκτιμήσεις του δικτύου)
x2_c1_test = linspace(min(X2_c1),max(X2_c1), 1000);
v2_c1_test = zeros(size(x2_c1_test));
for i = 1:length(x2_c1_test)
 z1 = W1 * x2_c1_test(i) + b1;
 a1 = max(0, z1); % ReLU ενεργοποίηση
 z2 = W2 * a1 + b2;
 v2_c1_test(i) = z2;
end

% Plot του κόστους
figure;
plot(1:epochs, costs, '-');
title('Cost per Epoch');
xlabel('Epoch');
ylabel('Cost (MSE)');

% Plot του v1(X) με σημεία
figure;
plot(x2_c1_test, omega(v2_c1_test), 'o');
title('v_1(X) vs X');
xlabel('X (S_t)');
ylabel('v_1(X)');

% Υπολογισμός και αποθήκευση των αποτελεσμάτων
omega_v2_c1_test = omega(v2_c1_test); % Υπολογισμός της συνάρτησης omega
save('v2_c1_results_test.mat', 'x2_c1_test', 'omega_v2_c1_test'); % Αποθήκευση δεδομένων
