%read_in_data

%Script to load in data from various output files from the IonWake/DRIAD
%code.  All datasets have directory path/file_prefix = datapath


%% Read in the parameters from param file
    fid = fopen([datapath 'params.txt']);
    while ~feof(fid)
        tline = fgetl(fid);
        %disp(tline);
        [~,q] = find(tline == '%');
        tempval = deblank(tline(1:q-1));
        tempvar = tline(q+2:end);
        eval([tempvar ' = ' tempval ';']);
    end
    fclose(fid);
    %% read in specific data from debug file
    fid = fopen([datapath 'debug.txt']);
    for i = 1:67 %skip first 64 lines
        tline = fgetl(fid);
    end
    for i = 1:5 %get BETA SIGMA RESX, RESZ, NUM_GRID_PTS
        tline = fgetl(fid);
        [~,q] = find(tline == ' ');
        tempvar = deblank(tline(1:q(1)-1));
        tempval = tline(q(end)+1:end);
        eval([tempvar ' = ' tempval ';']);
    end
    fclose(fid);
    
    %% Read in dust particle positions, vel, acc -- reshape
    try
    data = csvread([datapath 'dust-pos-trace.txt']);
    num_dust = 2;
    increment = num_dust+1;
    time = data(1:increment:end,1);
    data(1:increment:end,:) = [];
    
    positions = zeros(length(time),num_dust,3);
    acc = zeros(length(time),num_dust,3);
    vel = zeros(length(time),num_dust,3);
    for i = 1:length(time)
        range = (1:num_dust) + (i-1)*num_dust;
        positions(i,:,:) = data(range,1:3);
        vel(i,:,:) = data(range,4:6);
        acc(i,:,:) = data(range,7:9);
    end
    %set positions to match times density maps are output
    pos = positions(10:10:end,:,:);
    clear data
    catch
        disp(['no data in dust-pos-trace, dust not moving'])
        try
          pos = csvread([datapath 'dust-pos.txt']); 
        catch
            if dataset == 3
                disp('using pos for M = 1.0')
                pos = [0 0 0.0086062-BOX_CENTER];
            end
        end
    end
        
    %% Read in the charge
    q = csvread([datapath 'dust-charge.txt']);
    q(:,end) = []; %gets rid of last columns of zeros
    %% Read in grid location, ion potential, and ion density
    grid_data = csvread([datapath 'ion-den.txt']);
    
    %determine how many grid points there are
    num_pts = NUM_GRID_PTS;
    grid_pts = grid_data(1:num_pts,:);
    density =grid_data(num_pts+1:end,1);
    potential = grid_data(num_pts+1:end,2);
    
    X = reshape(grid_pts(:,1),RESX,RESZ);
    Z = reshape(grid_pts(:,3),RESX,RESZ);
    
    den = reshape(density,RESX,RESZ,round(length(density)/num_pts));
    pot = reshape(potential,RESX,RESZ,round(length(potential)/num_pts));
    clear grid_data