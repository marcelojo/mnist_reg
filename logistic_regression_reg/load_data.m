function [all_data, all_label, data, label] = load_data(all_data, train_qty)
    % Labels are the first columns
    all_label = all_data(:,1);

    % Copy data from second column
    all_data = all_data(:,2:end);

    % Just to be faster, we can train only in a part of the entire trainind dataset
    data = all_data(1:train_qty, :);
    label = all_label(1:train_qty);
endfunction