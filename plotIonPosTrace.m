% project: IonWake
% File Type: script - plot data
% File Name: plotIonPosTrace.m
% Git location: IonWake
%
% created: 7/25/2017
%
% Edits
%	Last Modified: 7/25/2017
%	Contributor(s):
%		Name: Dustin Sanford
%		Contact: Dustin_Sanford@baylor.edu
%       Last Contribution: 7/25/2017
%
% Description:
%   Plots the path of a single ion in 3D
%
% Input:
%   A text file with the name specified by the fileName
%   variable. Data should be in the format:
%       <x-pos>, <y-pos>, <z-pos>
%       <x-pos>, <y-pos>, <z-pos>
%       ...
%   there should be no spaces before the x-position or
%   after the z-position.
%
% Output:
%   A 3D plot of the ion position is created. The plot
%   is not automaticaly saved.
%

clear

% the file name that the ion trace data is stored in

% ...
fileName = 'ionPosTrace.txt';

% read in the data from the file
data = csvread(fileName);

% plot the positions
plot3(data(:,1),data(:,2),data(:,3), 'marker', 'o', 'LineWidth', 1,...
    'LineJoin', 'round');

% format the figure
set(gcf,'Color','w');
set(gca,'FontName','arial','FontSize',14,'YMinorTick','on','XMinorTick','on')
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Single Ion Trace');
grid on;
axis equal;