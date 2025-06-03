set(groot, 'DefaultAxesFontName', 'Jost');
set(groot, 'DefaultTextFontName', 'Figtree');

set(groot, 'DefaultFigureColor', '#2c2c2c');         #2c2c2c
set(groot, 'DefaultAxesXColor', [1 1 1]);          
set(groot, 'DefaultAxesYColor', [1 1 1]);         
set(groot, 'DefaultAxesColor', '#2c2c2c');           % Axes background white
set(groot, 'DefaultTextColor', [1 1 1]);           % Black text

% Load the data
% angles, ranges, convolved, disparity_mask, extended_ranges, smoothed_ranges
data = importdata('laser_scan_data.txt');
angles = data.data(:, 1);
ranges = data.data(:, 2);
convolved = data.data(:, 3);
disparity_mask = data.data(:, 4);
extended_ranges = data.data(:, 5);
smoothed_ranges = data.data(:, 6);

% Convert polar to Cartesian coordinates for plotting
x = ranges .* cos(angles);
y = ranges .* sin(angles);

x_ext = extended_ranges .* cos(angles);
y_ext = extended_ranges .* sin(angles);

% Plot the LiDAR scan
figure (1);
set(gcf, 'Color', '#2c2c2c');
scatter(x, y, 10, 'filled');
hold on
scatter(x_ext, y_ext, 10, 'filled');
axis equal;
xlabel('X (m)');
ylabel('Y (m)');
title('LiDAR LaserScan Data');
grid on;
print('lidar_scan_plot', '-dpng', '-r300');

% Plot the scan data straight across
figure (2);
scatter(angles, ranges, 10, 'filled', ...
       'MarkerFaceColor', '#aa55ff', ... 
       'MarkerEdgeColor', '#aa55ff');
ylim([0 6.0]);
xlim([-pi pi]);
xticks([-pi -pi/2 0 pi/2 pi]);
xticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'});
xlabel('Angle (rad)');
ylabel('Range (m)');
title('LIDAR SCAN DATA', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gcf, 'Color', '#2c2c2c');
set(gcf, 'InvertHardcopy', 'off');
print('unwrapped_scan.png', '-dpng', '-r300');

% Plot the convolved scan data and the detected disparities
disparity_indices = (1:length(ranges)) .* disparity_mask';
disparity_indices = disparity_indices(disparity_indices ~= 0);

figure (3);
h = stem(angles, convolved);
set(h, 'Marker', 'o', ...
       'MarkerSize', 1, ...
       'MarkerFaceColor', '#bd136e', ...  #bd136e
       'MarkerEdgeColor', '#bd136e', ...   % Make edge color red (same as fill)
       'Color', '#bd136e');
hold on
h_disp = scatter(angles(disparity_indices), 0*convolved(disparity_indices), 75, ...
    'MarkerFaceColor', '#e5db73', ...        % Yellow-ish
    'MarkerEdgeColor', [1 1 1], ...        % white
    'MarkerFaceAlpha', 0.25);
legend(h_disp, 'Identified Disparity Indices', 'Location', 'southeast', 'TextColor', [1 1 1]);
ylim([-0.1 0.1]);
yticks([]); % Remove y-axis ticks and markers
xlim([-pi/2 pi/2]);
xticks([-pi/2 -pi/4 0 pi/4 pi/2]);
xticklabels({'-\pi/2','-\pi/4','0','\pi/4','\pi/2'});
xlabel('Angle (rad)');
ylabel('Backwards Difference Result');
title('EDGE DETECTION RESULT', 'FontSize', 16, 'FontWeight', 'bold');
set(gcf, 'Color', '#2c2c2c');
set(gcf, 'InvertHardcopy', 'off');
print('convolved_scan.png', '-dpng', '-r300')


figure (4);
orig = scatter(angles, ranges, 1, 'filled', ...
       'MarkerFaceColor', '#e5db73', ... 
       'MarkerEdgeColor', '#e5db73');
hold on
% Only plot where extended_ranges ~= ranges
diff_idx = find(extended_ranges ~= ranges);
ext = scatter(angles(diff_idx), extended_ranges(diff_idx), 10, 'filled', ...
       'MarkerFaceColor', '#bd136e', ...    
       'MarkerEdgeColor', '#bd136e');
ylim([0 6.0]);
% xlim([-pi pi]);
% xticks([-pi -pi/2 0 pi/2 pi]);
% xticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'});
xlim([-pi/2 pi/2]);
xticks([-pi/2 -pi/4 0 pi/4 pi/2]);
xticklabels({'-\pi/2','-\pi/4','0','\pi/4','\pi/2'});
xlabel('Angle (rad)');
ylabel('Range (m)');
title('EXTENDED SCAN DATA', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gcf, 'Color', '#2c2c2c');
set(gcf, 'InvertHardcopy', 'off');
print('extended_scan.png', '-dpng', '-r300');

# Define the size of the kernel (odd number) and standard deviation
size = 5;
sigma = 1;

# Create the 1D Gaussian kernel
h = fspecial('gaussian', [size 1], sigma);
y = conv(smoothed_ranges, h, 'same');


figure (5);
ext = scatter(angles, extended_ranges, 5, 'filled', ...
       'MarkerFaceColor', '#aa55ff', ...    
       'MarkerEdgeColor', '#aa55ff');
hold on
% smoothed = plot(angles, smoothed_ranges, 'Color', '#e5db73', 'LineWidth', 2);
smoothed = plot(angles, y, 'Color', '#e5db73', 'LineWidth', 1);
ylim([0 6.0]);
% xlim([-pi pi]);
% xticks([-pi -pi/2 0 pi/2 pi]);
% xticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'});
xlim([-pi/2 pi/2]);
xticks([-pi/2 -pi/4 0 pi/4 pi/2]);
xticklabels({'-\pi/2','-\pi/4','0','\pi/4','\pi/2'});
xlabel('Angle (rad)');
ylabel('Range (m)');
grid on;
set(gcf, 'Color', '#2c2c2c');
set(gcf, 'InvertHardcopy', 'off');
print('smoothed_scan.png', '-dpng', '-r300');