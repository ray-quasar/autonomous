% Load the data
data = load('laser_scan_data.txt');
angles = data(:, 1);
ranges = data(:, 2);

% Convert polar to Cartesian coordinates for plotting
x = ranges .* cos(angles);
y = ranges .* sin(angles);

% Plot the LiDAR scan
figure;
scatter(x, y, 10, 'filled');
axis equal;
xlabel('X (m)');
ylabel('Y (m)');
title('LiDAR LaserScan Data');
grid on;