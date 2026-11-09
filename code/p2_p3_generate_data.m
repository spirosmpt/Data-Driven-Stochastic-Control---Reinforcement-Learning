% Parameters

N = 1000; % Number of actions
T = N + 1; % Number of states S1 to S1001

% Initialize S1 ~ N(0, 1)
S = zeros(T, 1);
S(1) = randn();

% Generate random actions (αt)
actions = randi([1, 2], N, 1);

% Generate Wt ~ N(0, 1)
W = randn(N, 1);

% Simulate S_t and calculate reward R(S_{t+1})
set_1 = []; % Pairs for α = 1
set_2 = []; % Pairs for α = 2

for t = 1:N
  if actions(t) == 1
    S(t+1) = 0.8 * S(t) + 1 + W(t);
  elseif actions(t) == 2
    S(t+1) = -2 + W(t);
  end

  % Calculate reward for the next state S_{t+1}
  % Ensure reward is in [0, 2]

  reward = min(2, S(t+1)^2);
  % Store pairs in the appropriate set

  if actions(t) == 1
    set_1 = [set_1; S(t), S(t+1), reward];
  elseif actions(t) == 2
    set_2 = [set_2; S(t), S(t+1), reward];
  end
end

% Save the sets as .mat files
save('Set1_Pairs.mat', 'set_1');
save('Set2_Pairs.mat', 'set_2');

disp('Data files with rewards saved as Set1_Pairs.mat and Set2_Pairs.mat');

% Plot the reward vs state transitions
figure;
scatter(set_1(:, 1), set_1(:, 3), 'r.', 'DisplayName', 'α = 1'); hold on;

scatter(set_2(:, 1), set_2(:, 3), 'b.', 'DisplayName', 'α = 2');
title('Reward vs State Transition');
xlabel('S_t');
ylabel('R(S_{t+1})');
legend;
grid on;
scatter(set_1(:, 1), set_1(:, 3));
title('Reward vs State S_t');
xlabel('S_t');
ylabel('R(S_{t+1})');
