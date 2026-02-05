clear
close all
%% dust position data
qe = -1.602e-19;

%path = 'chain40Pa/short_cylinder_low_temp/ch_40_';
%path = 'chain40Pa/Mach0_03/ch_40_';
%path = 'single_chain/Data_40Pa_Ti494/ch_40_'; 
%path = 'single_chain/Mar4_OldData/Data2_E100_40Pa_Ti290/ch_40_'; 
%path = 'single_chain/Mar4_OldData/Data3_E510_40Pa_Ti290/ch_403_'; 
%path = 'single_chain/Mar4_OldData/Data4_E1000_40Pa_Ti290/ch_40_'; 
%path = 'single_chain/Mar4_OldData/Data5_E245_40Pa_Ti612/ch_40_'; 

%path = 'single_chain/Data2_E100_Ti290/ch_40_'; 
%path = 'single_chain/Data3_E510_Ti290/ch_403_'; 
%path = 'single_chain/Data4_E1000_Ti290/ch_40_'; 
%path = 'single_chain/Data5_E245_Ti612/ch_40_'; 
%path = 'single_chain/Data5_E245_Ti464/ch_40_';
%path = 'single_chain/Data4_E1000_Ti290_Jun18/ch_40_';
%path = 'single_chain/Data5_E245_Ti464_Jun22/ch_40_';

 path = '/home/tycho_brahe/root/academic/research/data/IonWake/test/test_';
 axial_extent = 1.75;
radial_extent = 0.85;
%path = 'Mar19_N600/ch_40_';
% num_dust = 600;

%path = 'dust_N200/Jun6_Data4/ch_40_';
%path = 'dust_N200/Apr6_Data4/ch_40_';
%path = 'dust_N200/Apr8_Data2/ch_40_';
%path = 'dust_N200/Apr10_Data3/ch_40_';
%num_dust = 196;
% path = 'dust_N200/ch_40_';
% axial_extent = 3.2; %in mm
% radial_extent = 0.5; % in mm

%% read in specific data from debug file
% lines which have needed data from param file
%A=[35:39 41 42 47 48 50 51 53 54 63:69 71 72 75 81 82 84 85 91 95]; %with TEMP_GAS - new file
%Also need to change i = 1:122
A=[33 34 35 36 38 39 41 44 50 51 53 54 60:66 68 69 72 [78 79 80 81 82 84 85 92 116]+2];
fid = fopen([path 'debug.txt']);
for i = 1:118 %skip first N lines
    tline = fgetl(fid);
    if any(ismember(i,A))  

    [~,q] = find(tline == ' ');
    tempvar = deblank(tline(1:q(1)-1));
    tempval = tline(q(end)+1:end);
    eval([tempvar ' = ' tempval ';']);
    end
end

tline = fgetl(fid);
tline = fgetl(fid);
tline = fgetl(fid);
tline = fgetl(fid); %NUM_DUST on line 122
[~,q] = find(tline == ' ');
    tempvar = deblank(tline(1:q(1)-2));
    tempval = tline(q(end)+1:end);
    eval([tempvar ' = ' tempval ';']);
  
fclose(fid);
%% Dust position, velocity, acceleration data
data = csvread([path 'dust-pos-trace.txt']);
increment = NUM_DUST+1;
time = data(1:increment:end,1);
data(1:increment:end,:) = [];
positions = zeros(length(time),NUM_DUST,3);
acc = zeros(length(time),NUM_DUST,3);
vel = zeros(length(time),NUM_DUST,3);
for i = 1:length(time)-1
    range = (1:NUM_DUST) + (i-1)*NUM_DUST;
    positions(i,:,:) = data(range,1:3);
    vel(i,:,:) = data(range,4:6);
    acc(i,:,:) = data(range,7:9);
end

%%
figure
set(gcf,'Position',[ 488  342  908  172])
set(gcf,'Color','w')
%set(gcf, 'Color','w','Position',[85 250  475  100])
[a,~,c] = size(positions);
%correction since final positions read in as zeros at end of simulation
if a == 12000
    end_pos = csvread([path 'dust-pos.txt']);
    positions(12000,:,:) = end_pos;
end
incr = ceil(a/200);
i = 1;
plot3(positions(i,:,3)*1e3,positions(i,:,1)*1e3,positions(i,:,2)*1e3,'.','MarkerSize',8)
%axis off
view([0 0])
xlabel('z')
ylabel('x')
zlabel('y')
xlim([-1 1]*HT_CYL*1e3)
ylim([-1 1]*RAD_CYL*1e3)
zlim([-1 1]*RAD_CYL*1e3)
set(gca, 'nextplot','replacechildren')

% v = VideoWriter([path 'chain_bundle.avi']);
% v.Quality = 40;
% v.FrameRate = 30;
% open(v);

for i = [1 incr:incr:a]
    plot3(positions(i,:,3)*1e3,positions(i,:,1)*1e3,positions(i,:,2)*1e3,'.','MarkerSize',8)
    view([0 10])
    text(.7*HT_CYL*1e3,.5*RAD_CYL*1e3,.5*RAD_CYL*1e3,['t = ' num2str(i*1e-4,'%4.3f')]);
    pause(0.1)
%      frame = getframe(gcf);
%   writeVideo(v,frame)
end

pos = squeeze(positions(end-1,:,:));
%Now rotate about the z-axis
for theta = 0.1:0.1:2*pi
    pos2 = ([cos(theta) sin(theta) 0; -sin(theta) cos(theta) 0; 0 0 1]*pos')';
    plot3(pos2(:,3)*1e3,pos2(:,1)*1e3,pos2(:,2)*1e3,'r.','MarkerSize',8)
    view([0 10])
    text(.7*HT_CYL*1e3,.5*RAD_CYL*1e3,.5*RAD_CYL*1e3,['t = ' num2str(i*1e-4,'%4.3f')]);
    pause(0.1)
%      frame = getframe(gcf);
%   writeVideo(v,frame)
end
%close(v)
%% Plot some 100-micron slices
figure
set(gcf, 'Position',[488, 50,560 730])
a = 1; % a=1 for rotation about z, a=2 for slice at different x
pos = squeeze(positions(end-1,:,:));
for i = 1:5    
    if a == 1
    theta = 18*(i-1)*pi/180;
    pos = ([cos(theta) sin(theta) 0; -sin(theta) cos(theta) 0; 0 0 1]*pos')';
[p,~] = find(abs(pos(:,1)) < 50e-6);
    else
[p,~] = find(abs(((i-3)*75e-6)+pos(:,1)) < 50e-6);
    end
subplot(5,1,i)
set(gca,'Units','Normalized','Position',[.09 1-(i*.185) .9 .185])
plot(pos(p,3)*1e3, pos(p,2)*1e3,'.','MarkerSize',8)
if a == 1
    text(3, .3, ['\theta = ',num2str(theta*180/pi), '^o'])
else
text(2.75, .3, ['x = ',num2str(((i-3)*75)) ' \mum'])
end
xlim([-1 1]*HT_CYL*1e3)
ylim([-1 1]*.5*RAD_CYL*1e3)
end
xlabel('z (mm)')
ylabel('y (mm)')


%% step through the dust cluster in the y-direction
v = VideoWriter([path 'chain_bundle_slices.avi']);
v.Quality = 40;
v.FrameRate = 5;
open(v);
figure
set(gcf,'Position',[ 488  341  908  172])
rad_mult = ceil(max(sqrt(sum(pos(:,1:2).^2,2)))/RAD_CYL*10)/10;
l = 36;
center = rad_mult*RAD_CYL/l;
laser_hw = 50e-6; %half-width of the laser fan
laser_edge = [-1 -1 1 1]*laser_hw;
laser_top = [-1 1 1 -1]*rad_mult*RAD_CYL*1e3;
for i = 1:(2*l+1)
    x = (i-l)*center; %current x-location of "laser fan"

    [p,~] = find(abs(x-pos(:,1)) < laser_hw); %particles inside slice set by laser
    %scale the size of the point plotted by the distance from the center of
    %the laser slice
    s = (1 - abs(pos(p,1) - x)/laser_hw)*16 +1; 
  
     endview = axes('Units','normalized','Position',[.01 .01 .2 .98]);
        plot(pos(:,1,end)*1e3,pos(:,2,end)*1e3,'.')
        hold on
        plot([x x]*1e3, [-1 1]*rad_mult*RAD_CYL*1e3,'r')
        patch((x+laser_edge)*1e3, laser_top,'r','Facealpha',.2)
        xlim([-1 1]*rad_mult*RAD_CYL*1e3)
        ylim([-1 1]*rad_mult*RAD_CYL*1e3)
        text(-.1 ,.85*rad_mult, 'End view')
        set(endview,'XTick',[], 'YTick',[])
    sideview = axes('Units','normalized','Position',[.22 .01 .78 .98]);
scatter(pos(p,3,end)*1e3,pos(p,2,end)*1e3,s, 'Filled')  
text(3., .85*rad_mult, ['x = ',num2str((i-l)*5,'%3i') ' \mum'])
text(-.5 ,.85*rad_mult, 'Side view')
xlim([-1 1]*HT_CYL*1e3)
ylim([-1 1]*rad_mult*RAD_CYL*1e3)
set(sideview,'XTick',[], 'YTick',[])
%  frame = getframe(gcf);
%   writeVideo(v,frame)
end

close(v)

%%
% %% plot the acceleration
% figure
% %set(gcf, 'Color','w','Position',[85 250  475  100])
% i = 1;
% plot(positions(i,:,3)/HT_CYL,acc(i,:,3),'.','MarkerSize',8)
% hold on
% plot(positions(i,:,1)/RAD_CYL,acc(i,:,1),'.','MarkerSize',8)
% plot(positions(i,:,2)/RAD_CYL,acc(i,:,2),'.','MarkerSize',8)
% %axis off
% xlabel('pos')
% ylabel('acc')
% ylim([-100 100])
% xlim([-1 1])
% hold off
% set(gca,'NextPlot','replacechildren')
% incr = ceil(a/100);
% for i = 2:incr:length(time)-1
%     plot(positions(i,:,3)/HT_CYL,acc(i,:,3),'.','MarkerSize',8)
%     hold on
%     plot(positions(i,:,1)/RAD_CYL,acc(i,:,1),'.','MarkerSize',8)
%     plot(positions(i,:,2)/RAD_CYL,acc(i,:,2),'.','MarkerSize',8)
%     xlabel('pos')
%     ylabel('acc')
%     text(.75,65,['t = ' num2str(i*1e-4,'%4.3f')]);
%     ylim([-10 10])
%     xlim([-1 1])
%     hold off
%     pause(0.1)
% end

%% Plot the charge
q = csvread([path 'dust-charge.txt']);
qe = -1.602e-19;
[a,b]=size(q);
T = length(time);
if a > T
    q(end,:) = [];
end
q(:,end) = []; %gets rid of last columns of zeros
figure
plot(time, q/qe,'.')
%xlim([.06 .08])

avg_q = zeros(T - 100,NUM_DUST);
for i = 50:(T-50)
%range =((i-1)*50+1:i*50) + 25;
range = (i-49):(i+49);
avg_q(i-49,:) = mean(q(range,:))/qe;
end
figure
plot(time(50:(T-50)),avg_q)
xlabel('Time (s)')
ylabel('Charge (e^-)')
%%
for idx = [1000:2000:T T]
figure
set(gcf, 'Color','w', 'Position',[350 350 800 283])
%   adj_q = 4*pi*epsilon0*RAD_DUST*TeV*(1-RAD_DUST/DEBYE_I);

cmap=hot;
cmap(1:32,:) = [];
colormap(cmap)
%meanq = mean(q(1:floor(T/2),:));
%meanq = mean(q(floor(T/2):T,:));
meanq = mean(q((idx-100:idx),:));
CData2 = meanq/qe;
cmax = max(max(CData2));
cmin = min(min(CData2));
num_color = length(cmap);
color_charge2 = floor((CData2-cmin)/(cmax-cmin)*(num_color-1))+1;

pos = squeeze(positions(idx,:,:))*1e3;
scatter3(pos(:,3),pos(:,1),pos(:,2),25,cmap(color_charge2,:),'Filled')

view([0 90])
xlim([-1 1]*HT_CYL*1e3*0.8)
ylim([-1 1]*RAD_CYL*1e3)
text(0, 1.1*RAD_CYL*1e3,['Time = ' num2str(idx/1e4) ' s'])
title('Dust Charge (e^-)','Fontsize',10,'Position',[HT_CYL*.75e3,RAD_CYL*1.05e3, 0]);
xlabel('z (mm)')
ylabel('x (mm)')
dustplot = gca;
set(dustplot,'Color','k')
set(dustplot, 'Position', [.0379 .25 .861 .592])
%     set(dustplot,'NextPlot','replacechildren')
% hold on

cdiff = cmax-cmin;
caxis('Manual')
cax = colorbar('YTick',0:1/7:1,'YTickLabel',...
    {num2str(round(cmin/1e2)*1e2), num2str(round((cmin+cdiff*1/7)/1e2)*1e2),...
    num2str(round((cmin+cdiff*2/7)/1e2)*1e2),num2str(round((cmin+cdiff*3/7)/1e2)*1e2),...
    num2str(round((cmin+cdiff*4/7)/1e2)*1e2),num2str(round((cmin+cdiff*5/7)/1e2)*1e2),...
    num2str(round((cmin+cdiff*6/7)/1e2)*1e2),num2str(round(cmax/1e2)*1e2)});
set(cax,'Position',[.905 .25 .039 .586])
%export_fig([path 'dust_charge_t' num2str(idx/1e3)], '-jpg', '-r150')
end
%% Calculate pair correlation function g(r).  Uses positions (pos) from end of simulation.
%pos = pos/1e3;
gr_pair_correlation3
%export_fig([path 'g_r_central_chain'], '-jpg', '-r150')
gr_pos = (binwidth:binwidth:DEBYE*N)/.001;
save([path 'gr_data'],'gr','gr_pos','MACH','TEMP_ELC','TEMP_ION','meanq')
% %%
% CData = q/qe;
% cmax = max(max(CData));
% cmin = min(min(CData));
% num_color = length(cmap);
% colormap(cmap)
% color_charge = floor((CData-cmin)/(cmax-cmin)*(num_color-1))+1;
% figure
% ax = axes;
% for t = 1:incr:T-1
%     for i = 1: NUM_DUST
%         plot(ax,positions(t,i,3)*1e3,positions(t,i,1)*1e3,'ko','MarkerFaceColor',cmap(color_charge(t,i),:))
%         hold on
%     end
% %     if(t==321)
% %         pause(.1)
% %     end
%     hold off
%     text(.8 * HT_CYL*1e3, .8 * RAD_CYL*1e3, num2str(t))
%     xlim([-1 1]*HT_CYL*1e3)
%     ylim([-1 1]*RAD_CYL*1e3)
%     pause(0.01)
% end