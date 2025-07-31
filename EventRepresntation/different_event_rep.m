clc; clearvars; close all;

%filename = "Z:\slider_seg.npy";
filename = "Z:\Datasets\inputs\synthetic\EMS_479\event_threshold_0.1\gray_events_data.npy";
%filename = "Z:\Datasets\inputs\SEENIC\hubble-orbit-slow-ambient\event_threshold_0.1\gray_events_data.npy";

event_frame(filename)

event_accumulation(filename)

% Time surface variants
time_surface(filename, 0.05);

% Voxel grid with signed polarity
voxel = voxel_grid_interp(filename, 5, true);
plot_voxel_grid_3D(voxel, 0.1);

%saveAllFiguresToPath('C:\Users\Pierazhi\Documents\MATLAB\Tesi\eventRepresentation\Images\')


%%

function event_frame(npy_file)
    % Load events [x, y, t, p]
    events = readNPY(npy_file);
    x = round(events(:, 1)) + 1;
    y = round(events(:, 2)) + 1;
    p = events(:, 4);  % {0,1}

    % Convert polarity to {-1, +1}
    p_signed = 2 * p - 1;

    % Auto sensor size
    H = max(y);
    W = max(x);
    img = zeros(H, W);

    % Overwrite most recent polarity
    for i = 1:length(x)
        img(y(i), x(i)) = p_signed(i);
    end

    % Show result
    figure;
    imagesc(img);
    colormap gray;
    caxis([-1 1]);
    cb = colorbar;
    cb.Label.String = 'Last Event Polarity';
    axis image;
    title('Event Frame');
    xlabel('X [pixels]');
    ylabel('Y [pixels]');
end

function event_accumulation(npy_file)
    % Load events [x, y, t, p]
    events = readNPY(npy_file);
    x = round(events(:, 1)) + 1;
    y = round(events(:, 2)) + 1;
    p = events(:, 4);  % {0,1}

    % Convert polarity to {-1, +1}
    p_signed = 2 * p - 1;

    % Auto sensor size
    H = max(y);
    W = max(x);
    img = zeros(H, W);

    % Accumulate polarity
    for i = 1:length(x)
        img(y(i), x(i)) = img(y(i), x(i)) + p_signed(i);
    end

    % Show result
    figure;
    max_val = max(abs(img(:)));
    imagesc(img);
    colormap('gray');
    caxis([-max_val, max_val]);
    cb = colorbar;
    cb.Label.String = 'Net Brightness Change';
    axis image;
    title('Brightness Increment Image');
    xlabel('X [Pixels]');
    ylabel('Y [Pixels]');
end

function time_surface(npy_file, tau)
    % Load events: [x, y, t, p]
    events = readNPY(npy_file);
    x = round(events(:, 1)) + 1;
    y = round(events(:, 2)) + 1;
    t = events(:, 3);
    p = events(:, 4);  % polarity {0,1}
    p_signed = 2 * p - 1;

    H = max(y);
    W = max(x);
    t_ref = max(t);

    % Initialize time surfaces
    sae_both = zeros(H, W);
    sae_pos  = zeros(H, W);
    sae_neg  = zeros(H, W);
    sae_signed = zeros(H, W);
    sae_balance = zeros(H, W);

    for i = 1:length(x)
        decay = exp(-(t_ref - t(i)) / tau);
        if p(i) > 0
            sae_pos(y(i), x(i))  = decay;
            sae_signed(y(i), x(i)) = decay;
            sae_balance(y(i), x(i)) = sae_balance(y(i), x(i)) + decay;
        else
            sae_neg(y(i), x(i))  = decay;
            sae_signed(y(i), x(i)) = -decay;
            sae_balance(y(i), x(i)) = sae_balance(y(i), x(i)) - decay;
        end
        sae_both(y(i), x(i)) = decay;
    end

    % Plot all variants
    show_surface(sae_both, 'Time Surface');
    %show_surface(sae_pos, 'Time Surface (Positive)');
    %show_surface(sae_neg, 'Time Surface (Negative)');
    show_surface(sae_signed, 'Signed Time Surface');
    %show_surface(sae_balance, 'Polarity-Balanced Time Surface');
end

function show_surface(img, title_str)
    figure;
    imagesc(img);
    colormap('gray');
    cb = colorbar;
    if contains(title_str, 'Signed')
        cb.Label.String = 'Signed Event Recency';
    elseif contains(title_str, 'Time Surface')
        cb.Label.String = 'Event Recency (Exp Decay)';
    else
        cb.Label.String = 'Value';
    end
    axis image;
    title(title_str);
    xlabel('X [Pixels]');
    ylabel('Y [Pixels]');
end

function voxel = voxel_grid_interp(npy_file, B, signed)
    % Load events: [x, y, t, p]
    events = readNPY(npy_file);
    x = round(events(:, 1)) + 1;
    y = round(events(:, 2)) + 1;
    t = events(:, 3);
    p = events(:, 4);  % {0,1}
    p_signed = 2 * p - 1;

    H = max(y);
    W = max(x);
    voxel = zeros(B, H, W);

    t0 = min(t);
    t1 = max(t);
    dt = t1 - t0;
    t_norm = (t - t0) / dt * (B - 1);

    for i = 1:length(x)
        bin = t_norm(i);
        bin_low = floor(bin);
        bin_high = bin_low + 1;
        w_high = bin - bin_low;
        w_low = 1 - w_high;
        val = signed * p_signed(i) + ~signed * 1;  % signed or unsigned count

        if bin_low >= 0 && bin_low < B
            voxel(bin_low + 1, y(i), x(i)) = voxel(bin_low + 1, y(i), x(i)) + w_low * val;
        end
        if bin_high >= 0 && bin_high < B && bin_high ~= bin_low
            voxel(bin_high + 1, y(i), x(i)) = voxel(bin_high + 1, y(i), x(i)) + w_high * val;
        end
    end
end

function plot_voxel_grid_3D(voxel, threshold)
    % voxel: 3D array [B, H, W]
    % threshold: minimum absolute voxel value to show

    [B, H, W] = size(voxel);
    [t, y, x] = ind2sub(size(voxel), find(abs(voxel) > threshold));
    vals = voxel(sub2ind(size(voxel), t, y, x));

    % Normalize colors for polarity or intensity
    cmap_vals = (vals - min(vals)) / (max(vals) - min(vals));  % in [0,1]
    colors = repmat(cmap_vals, 1, 3);  % grayscale

    figure;
    scatter3(x, t, y, 10, colors, 'filled');
    xlabel('X [Pixels]'); ylabel('Time Bin'); zlabel('Y [Pixels]');
    title('3D Voxel Grid');
    view(20, 10);
    axis tight;
    grid on;
end

function saveAllFiguresToPath(userDefinedPath)
% SAVEALLFIGURESTOPATH Saves all open figures to a user-defined path using the plot title as filename.
%
%   saveAllFiguresToPath(userDefinedPath) saves all open MATLAB figures to the
%   specified userDefinedPath. Filenames are based on the axes title (from title()).
%   Spaces are replaced with underscores. Falls back to 'Figure_N' if no title is found.

% Input Validation
if nargin == 0
    error('User-defined path is required.');
end

if ~ischar(userDefinedPath) || isempty(userDefinedPath)
    error('User-defined path must be a non-empty string.');
end

if ~exist(userDefinedPath, 'dir')
    mkdir(userDefinedPath);
    disp(['Created directory: ' userDefinedPath]);
end

% List all open figures
openFigures = findall(0, 'Type', 'figure');

if isempty(openFigures)
    disp('No open figures found.');
    return
end

% Save figures
for i = 1:numel(openFigures)
    currentFigure = openFigures(i);
    axesHandles = findall(currentFigure, 'Type', 'axes');
    
    % Default fallback name
    fileName = sprintf('Figure_%d', i);
    
    % Try to extract the title from the first axes
    if ~isempty(axesHandles)
        titleText = get(get(axesHandles(1), 'Title'), 'String');
        
        if ischar(titleText) && ~isempty(strtrim(titleText))
            % Replace spaces with underscores and strip illegal characters
            fileName = regexprep(strtrim(titleText), '\s+', '_');
            fileName = regexprep(fileName, '[^\w]', '_');  % Replace non-word chars
        end
    end

    % Construct full path
    fullFilePath = fullfile(userDefinedPath, [fileName '.png']);

    % Save the figure
    try
        saveas(currentFigure, fullFilePath);
        disp(['Figure ' num2str(i) ' saved to: ' fullFilePath]);
    catch ME
        disp(['Error saving figure ' num2str(i) ': ' ME.message]);
    end
end
end

