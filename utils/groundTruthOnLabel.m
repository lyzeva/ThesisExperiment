%Test_label and Training_label must be scalar

function Groundtruth = groundTruthOnLabel(Test_label, Training_label)
Num_training = length(Training_label);
Num_test = length(Test_label);
Test_label = reshape(Test_label, Num_test, 1);
Training_label = reshape(Training_label, 1, Num_training);
a = repmat(Test_label, 1, Num_training);
b = repmat(Training_label, Num_test, 1);
Groundtruth = (a == b);

