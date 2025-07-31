clearvars; clc; close all;

% === Settings ===
blender_fix = false;

% === Load poses ===
gt = load_kitti_poses("Z:\Datasets\inputs\synthetic\EMS_100\traj.txt");
N = size(gt, 3);
gt_xyz = squeeze(gt(1:3, 4, :))';
 
% === Plot trajectory with pose indices
figure;
plot3(gt_xyz(:,1), gt_xyz(:,2), gt_xyz(:,3), 'k-', 'LineWidth', 2); hold on;

% Add a dot + label for each pose
for i = 1:10:N
    plot3(gt_xyz(i,1), gt_xyz(i,2), gt_xyz(i,3), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k');
    text(gt_xyz(i,1), gt_xyz(i,2), gt_xyz(i,3), ...
         sprintf('%d', i), ...
         'FontSize', 8, 'Color', 'r', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Trajectory with Pose Indices');
grid on; axis equal;

% === Translation Evolution with index labels (X, Y, Z)
labels = {'X', 'Y', 'Z'};
figure('Name', 'Translation Evolution with Indices');
for j = 1:3
    subplot(3,1,j);
    plot(1:N, gt_xyz(:,j), 'k-', 'LineWidth', 2); hold on;

    % Dots + labels
    for i = 1:10:N
        plot(i, gt_xyz(i,j), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k');
        text(i, gt_xyz(i,j), sprintf('%d', i), ...
             'FontSize', 8, 'Color', 'r', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end

    ylabel([labels{j} ' (m)']);
    grid on;
    if j == 1
        title('Trajectory Components with Pose Indices');
    end
    if j == 3
        xlabel('Pose Index');
    end
end




%% Functions

function poses = load_kitti_poses(filename)
    data = readmatrix(filename);
    [N, col] = size(data);

    if col == 12
        % KITTI format: 3x4 row-major
        poses = zeros(4, 4, N);
        for i = 1:N
            T = reshape(data(i, :), [3, 4]);
            poses(:,:,i) = [T; 0 0 0 1];
        end

    elseif col == 16
        % Full 4x4 matrix (flattened row-major): reshape then transpose
        poses = zeros(4, 4, N);
        for i = 1:N
            T = reshape(data(i, :), [4, 4])';  % <-- fix is here
            poses(:,:,i) = T;
        end

    else
        error('Unsupported pose format. Expected 12 (KITTI) or 16 (4x4) columns.');
    end
end



