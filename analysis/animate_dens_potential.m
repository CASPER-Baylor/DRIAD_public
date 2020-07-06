clear
close all
masterpath = '/home/tycho_brahe/root/academic/research/data/IonWake/';
datasets = ['test/test_'];

%% DUMMY 

NUM_GRID_PTS = 2048;
RESX = 32;
RESZ = 64;

%%

labels = 'abcd';
dkblue = [0.05 0.05 .6];
NUM_TIME_STEP = 12000;

for dataset = 1
    
    path = [masterpath deblank(datasets(dataset,:))];
    video_name = [masterpath datasets(dataset,1:4)];

    %% read in specific data from debug file
    % lines which have needed data from param file
    %%%A=[33 34 35 36 38 39 41 44 50:61 63 68:72 78 79 80 81 82 84 85 92 116];
    fid = fopen([path 'params.txt']);
    tline = fgetl(fid);
    while (tline ~= -1)
        [~,q] = find(tline == '%');
        tempval = deblank(tline(1:q(1)-1));    
        tempvar = deblank(tline(q(end)+1:end));    
        eval([tempvar ' = ' tempval ';']);Error in animate_dens_potential (line 237)

        tline = fgetl(fid);
    end
       
    %% read in dust position data and reshapeError in animate_dens_potential (line 237)

%   pos = csvread([path 'dust-pos.txt']); %static dust
    data = csvread([path 'dust-pos-trace.txt']); %moving dust
    
    increment = NUM_DUST+1;
    time = data(1:increment:end,1);
    data(1:increment:end,:) = [];
    
    positions = zeros(length(time),NUM_DUST,3);
    acc = zeros(length(time),NUM_DUST,3);
    vel = zeros(length(time),NUM_DUST,3);
    for i = 1:length(time)
        range = (1:NUM_DUST) + (i-1)*NUM_DUST;
        positions(i,:,:) = data(range,1:3);
        vel(i,:,:) = data(range,4:6);
        acc(i,:,:) = data(range,7:9);
    end
    pos = positions; %%% (10:10:end,:,:);
    
    %% read in data for the ion density and ion potential
    grid_data = csvread([path 'ion-den.txt']);
    
    %determine how many grid points there are
    HT_CYL_DEBYE = HT_CYL/DEBYE;
    RAD_CYL_DEBYE = RAD_CYL/DEBYE;
    mult = floor(HT_CYL_DEBYE/(RAD_CYL_DEBYE/1.01));
    num_pts = NUM_GRID_PTS;
    grid_pts = grid_data(1:num_pts,:);
    density =grid_data(num_pts+1:end,1);
    potential = grid_data(num_pts+1:end,2);
    
    X = reshape(grid_pts(:,1),RESX,RESZ);
    Z = reshape(grid_pts(:,3),RESX,RESZ);
    
    den = reshape(density,RESX,RESZ,round(length(density)/num_pts));
    pot = reshape(potential,RESX,RESZ,round(length(potential)/num_pts));
    %% Determine potential from outside ions
    %electric field
    Ex = P10X*X + P12X*X.*Z.^2 + P14X*X.*Z.^4;
    Ez = P01Z*Z + P03Z*Z.^3 + P05Z*Z.^5 + P21Z*X.^2.*Z + P23Z*X.^2.*Z.^3;
    %potential is -Integral(E dot ds)
    pot_outside = -(0.5*X.^2.*(P10X+P12X*Z.^2+P14X*Z.^4)...
        +0.5*(Z.^2.*(P01Z+P21Z*X.^2))...
        +0.25*Z.^4.*(P03Z + P23Z*X.^2) + (P05Z*Z.^6)/6);
    %% animation for density
    [~,~,cc]=size(den);
    cmap = colormap(jet);
    cmap2 = gray(32);
    lev = linspace(1.0,3.5,64); %levels for contour lines
    %final_den = squeeze(mean(den(:,:,(cc-49):cc),3)); %average of 50 maps
    final_den = squeeze(mean(den,3)); %average of all maps
    figure(1)
    set(gcf,'Position',[350  350  750 295],'color','w')
    colormap jet
    plt = axes;
    plt.Position =[.088 .103 .775 .815];
    contourf(plt,Z*1e3,X*1e3,final_den*SUPER_ION_MULT/DEN_FAR_PLASMA,lev,...
        'Linestyle','none');
    hold on
    c = colorbar;
    c.Label.String = 'n_i/n_{i0}';
    c.Position = [.8859 .1767 .038 .6642];
    axis equal
    hold on
    final_pos = reshape(squeeze(pos(cc,:,:)),[1,3]);
    col = round(abs(final_pos(:,2))/(0.55*RAD_CYL)*31)+1;
    col(col>32) = 32;
    scatter(final_pos(:,3)*1e3,final_pos(:,1)*1e3,40,cmap2(col,:),'.')
    ylim([-1 1])
    xlim([-3 3])
    ylabel('x (mm)')
    xlabel('z (mm)')
    title(['Ion Density        Time = ' num2str(cc*10*1e-4,'%4.3f')])
    plt.Color = cmap(1,:);
    %export_fig([video_name 'ion_density'], '-png', '-r150')
    %%
    plt.NextPlot = 'replacechildren';
    vid = VideoWriter([video_name 'density.avi']);
    vid.FrameRate = 15;
    open(vid);
    
    for i = [10:10:length(density)/num_pts]
        %final_den = squeeze(den(:,:,i));
        final_den = mean(den(:,:,i-9:i),3);
        contourf(plt,Z*1e3,X*1e3,final_den*SUPER_ION_MULT/DEN_FAR_PLASMA,lev,'Linestyle','none');
       %plt.Color = cmap(1,:);
       title(['Ion Density        Time = ' num2str(i*10*1e-4,'%4.3f')])
        hold on
        avg_pos = reshape( squeeze(pos(i-4,:,:)) , [1,3]);
        col = round(abs(avg_pos(:,2))/(0.55*RAD_CYL)*31)+1;
    col(col>32) = 32;
    scatter(avg_pos(:,3)*1e3,avg_pos(:,1)*1e3,40,cmap2(col,:),'.')
        axis equal
        %hold off
        frame = getframe(gcf);
        writeVideo(vid,frame)
        if i ==1200
            writeVideo(vid,frame)
            writeVideo(vid,frame)
            writeVideo(vid,frame)
            writeVideo(vid,frame)
            writeVideo(vid,frame)
        end
    end
    close(vid)
    
    %% read in dust charge
    CHARGE_DUST = csvread([path 'dust-charge.txt']);
    CHARGE_DUST(:,end) = [];
    CHARGE_DUST(end,:) = [];
    qe = -1.602e-19;
   
    %%
    figure
    qtime = (0:NUM_TIME_STEP)*1e-4;
    [p,~]= size(CHARGE_DUST);
    plot(qtime(1:p),CHARGE_DUST/-1.602e-19)
    xlabel('Time (s)')
    ylabel('charge (e)')
    title('Dust charge')
    %% Potential due to dust
    epsilon0 = 8.85e-12;
    POS1=repmat(reshape(grid_pts,[num_pts,1,3]),[1,NUM_DUST,1]);
    %each row of POS2 is a repeat of the first one.
    POS2=repmat(reshape(squeeze(pos(end,:,:)),[1,NUM_DUST,3]),[num_pts,1,1]);
    %find the distance between every grid point and dust position
    %dist_pts_inv is number_of_points x numDust in size
    dist_pts=sqrt(sum((POS1-POS2).^2,3)); %distance to a point itself is 0
    meanq = mean(CHARGE_DUST(:,:)); %%%
    [dim1, dim2] = size(meanq);
    if dim1 == 1
        V_dust = repmat(meanq,num_pts,1)/(4*pi*epsilon0)./dist_pts;
    else
        V_dust = repmat(meanq',num_pts,1)/(4*pi*epsilon0)./dist_pts;
    end
    V_dust = sum(V_dust,2);
    V_dust = reshape(V_dust,RESX,RESZ);
    %%  dust + ion potential
    if dataset == 4
        v_adj = 2.68;
        v = -.15:(.15+.15)/63:.15;
    else
        v_adj = 3.1;
        v = -.15:(.15+.15)/63:.15;
    end
incr = 10;
    figure(4)
    hold on
    set(gcf,'Position',[100  100 900 320],'color','w')
    cmap = colormap(jet);
    plt = axes;
    plt.Position = [0.0546    0.1370    0.7750    0.7667];
    contour(plt,Z*1e3,X*1e3,mean(pot(:,:,cc-(incr-1):cc),3)+pot_outside+V_dust-v_adj,v,'Fill','on')
    plt.Color = cmap(1,:);
    title(['Potential        Time = ' num2str(cc*10*1e-4,'%4.3f')])
    axis equal
    xlim([-1.8 1.8])
    ylim([-.5 .5])
    xlabel('z (mm)')
    ylabel('x (mm)')
    c = colorbar;
    c.FontSize = 10;
    c.Position = [0.8556    0.2155    0.0340    0.61406];
    caxis([min(v) max(v)])
    c.Label.String = '\Phi (V)';
    
    hold on
    dot_size = 20-18*abs(squeeze(pos(end,:,2)))/100e-6;
    dot_size(dot_size<2) = 2;
    scatter(squeeze(pos(cc,:,3))*1e3, squeeze(pos(cc,:,1))*1e3, dot_size,'k','filled')
   % export_fig([video_name 'potential'], '-png', '-r150')
    %%
    plt.NextPlot = 'replacechildren';
    vid = VideoWriter([video_name 'dustpotential.avi']);
    %vid.Quality = 40;
    vid.FrameRate = 3;
    open(vid);
    
    POS1=repmat(reshape(grid_pts,[num_pts,1,3]),[1,NUM_DUST,1]);
    incr = 10;
    for i = [10:incr:length(density)/num_pts]
        %each row of POS2 is a repeat of the first one.
        POS2=repmat(reshape(squeeze(pos(i,:,:)),[1,NUM_DUST,3]),[num_pts,1,1]);
        %find the distance between every grid point and dust position
        %dist_pts_inv is number_of_points x numDust in size
        dist_pts=sqrt(sum((POS1-POS2).^2,3)); %distance to a point itself is 0
        meanq = mean(CHARGE_DUST(:,:));%%%
        V_dust = repmat(meanq,num_pts,1)/(4*pi*epsilon0)./dist_pts;
        V_dust = sum(V_dust,2);
        V_dust = reshape(V_dust,RESX,RESZ);
        
        contourf(plt,Z*1e3,X*1e3,mean(pot(:,:,i-(incr-1):i),3)+pot_outside+V_dust-v_adj,...
            v,'Linestyle','none')
        %colorbar
        hold on
        dot_size = 20-18*abs(squeeze(pos(i,:,2)))/100e-6;
        dot_size(dot_size<2) = 2;
        scatter(squeeze(pos(i,:,3))*1e3, squeeze(pos(i,:,1))*1e3, dot_size,'k','filled')
        title(['Potential        Time = ' num2str(i*10*1e-4,'%4.3f')])
        frame = getframe(gcf);
        disp('')
        writeVideo(vid,frame)
        if i ==1200
            writeVideo(vid,frame)
            writeVideo(vid,frame)
            writeVideo(vid,frame)
            writeVideo(vid,frame)
            writeVideo(vid,frame)
        end
    end
    close(vid)
    close all
end%loop over datasets


