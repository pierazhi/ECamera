clearvars; clc; close all;

% === Settings ===
blender_fix = false;

% === Load poses ===
gt = load_kitti_poses("Z:\Datasets\outputs\SEENIC_appraoch-slow-ambient\demo_room\BA\pose_file_gt_BA_f044_whole.txt");
est = load_kitti_poses("Z:\Datasets\outputs\SEENIC_appraoch-slow-ambient\demo_room\BA\pose_file_est_BA_f044_whole.txt");
N = size(est, 3);

% === Apply Blender convention fix (flip camera Z) ===
if blender_fix
    R_flipZ = diag([1 1 -1]);
    for i = 1:N
        gt(1:3,1:3,i)  = gt(1:3,1:3,i) * R_flipZ;
        est(1:3,1:3,i) = est(1:3,1:3,i) * R_flipZ;
    end
end

% === Align only positions using Umeyama ===
gt_xyz = squeeze(gt(1:3, 4, :));
est_xyz = squeeze(est(1:3, 4, :));
[R_u, t_u, s_u] = umeyama_alignment(est_xyz, gt_xyz, true);
est_traj_aligned = zeros(4, 4, N);
for i = 1:N
    T = est(:,:,i);
    est_traj_aligned(1:3,1:3,i) = s_u * R_u * T(1:3,1:3);
    est_traj_aligned(1:3,4,i)   = s_u * R_u * T(1:3,4) + t_u;
    est_traj_aligned(4,4,i)     = 1;
end

% === First-pose orientation alignment ===
T_align = gt(:,:,1) * inv(est(:,:,1));  % only rotation correction
est_orient_aligned = zeros(4,4,N);
for i = 1:N
    est_orient_aligned(:,:,i) = T_align * est(:,:,i);
end

% === Compute scene scale and set quiver size ===
scene_min = min(gt_xyz, [], 2);
scene_max = max(gt_xyz, [], 2);
scene_diag = norm(scene_max - scene_min);
quiver_scale = 0.05 * scene_diag;

% === Compute translation error ===
gt_xyz_vec  = gt_xyz';  % Nx3
est_xyz_vec = squeeze(est_traj_aligned(1:3, 4, :))';
trans_error = sqrt(sum((gt_xyz_vec - est_xyz_vec).^2, 2));

% === Plot trajectories ===
figure;
plot3(gt_xyz_vec(:,1), gt_xyz_vec(:,2), gt_xyz_vec(:,3), 'k-', 'LineWidth', 2); hold on;
plot3(est_xyz_vec(:,1), est_xyz_vec(:,2), est_xyz_vec(:,3), 'm--', 'LineWidth', 2);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Trajectory Comparison');
legend('Ground Truth', 'Estimated');
grid on;

% === Plot translation error ===
figure;
plot(trans_error * 100, 'LineWidth', 2);
xlabel('Frame Index'); ylabel('Translation Error (cm)');
title('Absolute Translation Error per Frame');
grid on;

% === RMSE & per-axis stats ===
rmse = sqrt(mean(trans_error.^2));
fprintf('Translation RMSE: %.6f cm\n', rmse*100);

fprintf('Max Translation Error: %.6f cm\n', max(trans_error)*100);
fprintf('Mean Translation Error: %.6f cm\n', mean(trans_error)*100);

diff_xyz_cm = (gt_xyz_vec - est_xyz_vec) * 100;
rmse_xyz = sqrt(mean(diff_xyz_cm.^2, 1));
max_xyz  = max(abs(diff_xyz_cm), [], 1);

labels = {'X', 'Y', 'Z'};
for i = 1:3
    fprintf('\n[%s axis]\n', labels{i});
    fprintf('  RMSE:   %.3f cm\n', rmse_xyz(i));
    fprintf('  Max:    %.3f cm\n', max_xyz(i));
end

% === Per-axis evolution plots ===
for j = 1:3
    figure;
    plot(gt_xyz_vec(:,j), 'k-', 'LineWidth', 2); hold on;
    plot(est_xyz_vec(:,j), 'm--', 'LineWidth', 2);
    xlabel('Frame Index'); ylabel([labels{j} ' (m)']);
    title(['Translation Evolution: ' labels{j}]);
    legend('Ground Truth', 'Estimated');
    grid on;
end

% === Orientation Axis Plot ===
figure; hold on; axis equal; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Orientation Axes with First-Pose Alignment');
plot3(gt_xyz_vec(:,1), gt_xyz_vec(:,2), gt_xyz_vec(:,3), 'k-', 'LineWidth', 2);
plot3(est_xyz_vec(:,1), est_xyz_vec(:,2), est_xyz_vec(:,3), 'm--', 'LineWidth', 2);
axis_colors = {'r', 'g', 'b'};
step = max(1, round(N / 10));
for i = 1:step:N
    origin_gt = gt(1:3,4,i);
    origin_est = est_traj_aligned(1:3,4,i);
    for a = 1:3
        % Normalize and scale axis arrows
        dir_gt = quiver_scale * (gt(1:3,a,i) / norm(gt(1:3,a,i)));
        dir_est = quiver_scale * (est_orient_aligned(1:3,a,i) / norm(est_orient_aligned(1:3,a,i)));

        quiver3(origin_gt(1), origin_gt(2), origin_gt(3), ...
                dir_gt(1), dir_gt(2), dir_gt(3), ...
                'Color', axis_colors{a}, 'LineWidth', 1.2, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
        quiver3(origin_est(1), origin_est(2), origin_est(3), ...
                dir_est(1), dir_est(2), dir_est(3), ...
                'Color', axis_colors{a}, 'LineStyle', '--', ...
                'LineWidth', 1.2, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
    end
end
legend('GT', 'Estimated');

% === Orientation errors ===
rot_err_rpy   = zeros(N,3);
rpy_gt = zeros(N,3);
rpy_est = zeros(N,3);
for i = 1:N
    R_gt = gt(1:3,1:3,i);
    R_est = est_orient_aligned(1:3,1:3,i);
    rpy_gt(i,:)  = rotm2eul(R_gt, 'ZYX');
    rpy_est(i,:) = rotm2eul(R_est, 'ZYX');
end

% === Unwrap angles along time axis ===
rpy_gt = rad2deg(unwrap(rpy_gt));   % [N x 3], unwrapped and converted to deg
rpy_est = rad2deg(unwrap(rpy_est)); % same

% === Compute RPY error (angle difference) ===
rot_err_rpy = wrapTo180(rpy_est - rpy_gt);  % wrap to [-180,180] deg

% === RPY Evolution Plot ===

labels = {'Yaw (Z)', 'Pitch (Y)', 'Roll (X)'};
figure('Name', 'RPY (ZYX) evolution');
for j = 1:3
    subplot(3,1,j);
    plot(rpy_gt(:,j), 'k-', 'LineWidth', 2); hold on;
    plot(rpy_est(:,j), 'm--', 'LineWidth', 2);
    ylabel([labels{j} ' (°)']); xlabel('Frame Index');
    title(['RPY Evolution: ' labels{j}]);
    legend('Ground Truth', 'Estimated');
    grid on;
end

labels = {'Yaw (Z)', 'Pitch (Y)', 'Roll (X)'};
rpy_rmse = sqrt(mean(rot_err_rpy.^2, 1));
rpy_mean = mean(abs(rot_err_rpy), 1);  % abs in case bias flips
rpy_max  = max(abs(rot_err_rpy), [], 1);

fprintf('\n📐 Rotation Errors per Axis (deg):\n');
for i = 1:3
    fprintf('[%s]\n', labels{i});
    fprintf('  RMSE: %.3f°\n', rpy_rmse(i));
    fprintf('  Mean: %.3f°\n', rpy_mean(i));
    fprintf('  Max:  %.3f°\n\n', rpy_max(i));
end

% === Orientation Error: Angular Difference (deg) ===
ang_diff = zeros(N, 1);
for i = 1:N
    R_gt = gt(1:3,1:3,i);
    R_est = est_orient_aligned(1:3,1:3,i);
    R_err = R_gt' * R_est;  % relative rotation
    theta_rad = acos( max(-1, min(1, (trace(R_err) - 1) / 2)) );  % clip for numerical safety
    ang_diff(i) = rad2deg(theta_rad);
end

% === Plot angular difference over time
figure;
plot(ang_diff, 'LineWidth', 2);
xlabel('Frame Index'); ylabel('Angular Error (deg)');
title('Orientation Error (Angular Difference)');
grid on;

% === Print angular stats
fprintf('🔁 Angular Difference (all axes combined):\n');
fprintf('  RMSE: %.3f°\n', sqrt(mean(ang_diff.^2)));
fprintf('  Mean: %.3f°\n', mean(ang_diff));
fprintf('  Max:  %.3f°\n\n', max(ang_diff));

%% Function

function poses = load_kitti_poses(filename)
    data = readmatrix(filename);
    N = size(data, 1);
    poses = zeros(4, 4, N);
    for i = 1:N
        T = reshape(data(i, :), [4, 3])';
        T = [T; 0 0 0 1];
        poses(:, :, i) = T;
    end
end

function [R, t, c] = umeyama_alignment(x, y, with_scale)
%UMEYAMA_ALIGNMENT Least-squares similarity transformation between two point sets
%   [R, t, c] = umeyama_alignment(x, y, with_scale)
%   x and y: m x n matrices (m = dimensions, n = points)
%   with_scale: true/false, whether to estimate scale
%
%   Returns:
%     R: rotation matrix (m x m)
%     t: translation vector (m x 1)
%     c: scale factor (scalar)

    if nargin < 3
        with_scale = false;
    end
    
    if ~isequal(size(x), size(y))
        error('Input matrices x and y must be of the same size');
    end
    
    [m, n] = size(x); % m = dimensions, n = number of points
    
    % Compute means
    mean_x = mean(x, 2); % column vector
    mean_y = mean(y, 2);
    
    % Variance of x
    sigma_x = (1/n) * sum(vecnorm(x - mean_x, 2, 1).^2);
    
    % Covariance matrix
    cov_xy = zeros(m, m);
    for i = 1:n
        cov_xy = cov_xy + (y(:,i) - mean_y) * (x(:,i) - mean_x)';
    end
    cov_xy = cov_xy / n;
    
    % SVD
    [U, D, V] = svd(cov_xy);
    
    % Handle degenerate case
    % if rank(D) < m - 1
    %     error('Degenerate covariance rank, Umeyama alignment is not possible');
    % end
    
    % Construct S matrix
    S = eye(m);
    if det(U) * det(V) < 0
        S(end, end) = -1;
    end
    
    % Rotation
    R = U * S * V';
    
    % Scale
    if with_scale
        c = trace(D * S) / sigma_x;
    else
        c = 1.0;
    end
    
    % Translation
    t = mean_y - c * R * mean_x;

end


