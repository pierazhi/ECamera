clc; close all; clearvars;

% === Load Ground Truth ===
load('0013_groundtruth.mat');  % Contains: poses (Nx4x4), timestamps (Nx1), velocities ((N-1)x3), positions (Nx3)

timestamps = timestamps(:);    % Ensure column vector
N = size(poses, 1);

% === Compute overall frame frequency ===
dt_all = diff(timestamps);      % all time intervals
mean_dt = mean(dt_all);         % average time between timestamps
freq_hz = 1 / mean_dt;          % frequency in Hz

fprintf('🕒 Mean time step: %.6f s (%.2f Hz)\n', mean_dt, freq_hz);

% === Define Parameters ===
T_total = timestamps(end);  % Optional, used only if timestamps aren't regular
start_idx = 50;             % <-- You must set this
est = load_kitti_poses("Z:\Datasets\outputs\ELOPE_0013\demo_room\global_BA\pose_file_est_BA_f069_whole.txt");
N_est = size(est, 3);  % inferred from file
end_idx = start_idx + N_est;

% === Ground Truth Analysis ===
% 1. Plot 3D trajectory
figure;
plot3(positions(:,1), positions(:,2), positions(:,3), 'k-', 'LineWidth', 1.5); hold on;
plot3(positions(start_idx:end_idx,1), positions(start_idx:end_idx,2), positions(start_idx:end_idx,3), 'r-', 'LineWidth', 2.5);
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('Ground Truth Trajectory (with Estimated Segment Highlighted)');
legend('Full GT', 'Compared Segment'); grid on; 

% 2. Plot velocity
figure;
subplot(3,1,1);
plot(1:N-1, velocities(:,1), 'k'); hold on;
plot(start_idx:end_idx-1, velocities(start_idx:end_idx-1,1), 'r', 'LineWidth', 2);
ylabel('v_x [m/s]'); grid on;

subplot(3,1,2);
plot(1:N-1, velocities(:,2), 'k'); hold on;
plot(start_idx:end_idx-1, velocities(start_idx:end_idx-1,2), 'r', 'LineWidth', 2);
ylabel('v_y [m/s]'); grid on;

subplot(3,1,3);
plot(1:N-1, velocities(:,3), 'k'); hold on;
plot(start_idx:end_idx-1, velocities(start_idx:end_idx-1,3), 'r', 'LineWidth', 2);
xlabel('Frame Index'); ylabel('v_z [m/s]');
title('Velocity Components with Segment Highlighted'); grid on;

% 3. Altitude
altitude = abs(positions(:,3));
figure;
plot(1:N, altitude, 'k-'); hold on;
plot(start_idx:end_idx, altitude(start_idx:end_idx), 'r', 'LineWidth', 2);
xlabel('Frame Index'); ylabel('|Z| [m]');
title('Altitude with Estimated Segment Highlighted');
grid on;

% 4. Angular velocity
euler_angles = zeros(N, 3); angular_velocity = zeros(N-1, 3);
for i = 1:N
    R = squeeze(poses(i,1:3,1:3));
    euler_angles(i,:) = rotm2eul(R, 'XYZ');
end
for i = 1:N-1
    R0 = squeeze(poses(i,1:3,1:3));
    R1 = squeeze(poses(i+1,1:3,1:3));
    R_rel = R0' * R1;
    axang = rotm2axang(R_rel);
    angular_velocity(i,:) = axang(1:3) * axang(4) / (timestamps(i+1) - timestamps(i));
end

% figure;
% plot(1:N-1, angular_velocity, 'LineWidth', 1.5);
% xlabel('Frame Index');
% ylabel('Angular Velocity [rad/s]');
% legend('\omega_x', '\omega_y', '\omega_z');
% title('Angular Velocity Over Time'); grid on;

% 5. Euler angles
figure;
plot(1:N, rad2deg(euler_angles), 'k-'); hold on;
plot(start_idx:end_idx, rad2deg(euler_angles(start_idx:end_idx,:)), 'LineWidth', 2);
xlabel('Frame Index'); ylabel('Euler Angles [deg]');
legend('Full Roll', 'Full Pitch', 'Full Yaw', 'Seg Roll', 'Seg Pitch', 'Seg Yaw');
title('Euler Angles with Segment Highlighted'); grid on;

% === Estimated Trajectory Comparison ===
close all; 

% === Settings ===
blender_fix = false;

% === Load poses ===
gt = load_kitti_poses("Z:\Datasets\outputs\ELOPE_0013\demo_room\global_BA\pose_file_gt_BA_f069_whole.txt");
est = load_kitti_poses("Z:\Datasets\outputs\ELOPE_0013\demo_room\global_BA\pose_file_est_BA_f069_whole.txt");
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

% === RMSE & per-axis stats ===
rmse = sqrt(mean(trans_error.^2));
fprintf('Translation RMSE: %.6f m\n', rmse);
fprintf('Max Translation Error: %.6f m\n', max(trans_error));
fprintf('Mean Translation Error: %.6f m\n', mean(trans_error));

diff_xyz = (gt_xyz_vec - est_xyz_vec);
rmse_xyz = sqrt(mean(diff_xyz.^2, 1));
max_xyz  = max(abs(diff_xyz), [], 1);
labels = {'X', 'Y', 'Z'};
for i = 1:3
    fprintf('\n[%s axis]\n', labels{i});
    fprintf('  RMSE:   %.3f m\n', rmse_xyz(i));
    fprintf('  Max:    %.3f m\n', max_xyz(i));
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

dt = diff(timestamps(start_idx:end_idx-1));                 % [N-1 x 1]
est_vel = diff(est_xyz_vec) ./ dt;  % dt is [N-1 x 1]
gt_vel_used = velocities(start_idx : end_idx - 2, :);
z_gt = abs(positions(start_idx:end_idx-2, 3));     % also fix this to match
diff_vel = est_vel - gt_vel_used;
rms_vel_err = sqrt(sum(diff_vel.^2, 2));           % norm of velocity error per frame
norm_vel_err = rms_vel_err ./ z_gt;                % normalized by altitude
E_i = mean(norm_vel_err);                          % final score
fprintf('📉 Velocity Score E(i): %.6f\n', E_i);

labels = {'v_x', 'v_y', 'v_z'};
for d = 1:3
    figure;
    plot(gt_vel_used(:,d), 'k-', 'LineWidth', 1.5); hold on;
    plot(est_vel(:,d), 'r--', 'LineWidth', 1.5);
    xlabel('Frame Index'); ylabel([labels{d} ' [m/s]']);
    legend('Ground Truth', 'Estimated');
    title(['Velocity Comparison: ' labels{d}]);
    grid on;
end

% === Compute total path length for GT and Estimated ===
gt_deltas  = diff(gt_xyz_vec);                        % [N-1 x 3]
est_deltas = diff(est_xyz_vec);                       % [N-1 x 3]

gt_path_length  = sum(sqrt(sum(gt_deltas.^2, 2)));    % scalar
est_path_length = sum(sqrt(sum(est_deltas.^2, 2)));   % scalar

fprintf('GT path length:  %.3f m\n', gt_path_length);
fprintf('Est path length: %.3f m\n', est_path_length);

% === RMSE normalized by GT path length (in %) ===
rmse_norm_pct = (rmse / gt_path_length) * 100;
fprintf('Normalized RMSE w.r.t. path length: %.2f %%\n', rmse_norm_pct);

%% === Umeyama Alignment Function ===

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
    if rank(D) < m - 1
        error('Degenerate covariance rank, Umeyama alignment is not possible');
    end
    
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