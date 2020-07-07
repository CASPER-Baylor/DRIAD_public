%read in parameters

fid = fopen([path 'params.txt']);
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
    fid = fopen([path 'debug.txt']);
    for i = 1:112 %skip first 112 lines
        tline = fgetl(fid);
    end
    for i = 1:5 %get BETA RESX, RESz, NUM_GRID_PTS
        tline = fgetl(fid);
        [~,q] = find(tline == ' ');
        tempvar = deblank(tline(1:q(1)-1));
        tempval = tline(q(end)+1:end);
        eval([tempvar ' = ' tempval ';']);
    end
    fclose(fid);
%%
data = csvread([path 'dust-pos-trace.txt']);
increment = num_dust+1;
time = data(1:increment:end,1);
data(1:increment:end,:) = [];
%%
positions = zeros(length(time),num_dust,3);
acc = zeros(length(time),num_dust,3);
vel = zeros(length(time),num_dust,3);
for i = 1:length(time)
    range = (1:num_dust) + (i-1)*num_dust;
    positions(i,:,:) = data(range,1:3);
    vel(i,:,:) = data(range,4:6);
    acc(i,:,:) = data(range,7:9);
end
%get positions relative to the top particle
rel_pos = positions - repmat(positions(:,1,:),1,2,1);

%% Read in charge data
q = csvread([path 'dust-charge.txt']);
qe = -1.602e-19;
[a,b]=size(q);
T = length(time);
if a > T
    q(T+1:end,:) = [];
end
q(:,end) = []; %gets rid of last columns of zeros


avg_q = zeros(T,2);
avg_t = 50;%upstream and downstream timesteps for averaging charge
avg_q(1:(avg_t-1),1) = mean(q(1:(avg_t-1),1));
avg_q(1:(avg_t-1),2) = mean(q(1:(avg_t-1),2));
for i = avg_t:(T-avg_t-1)
    %range =(i-1)*50+1:i*50 + 25;
    range = (1:2*avg_t) + i-avg_t;
    avg_q(i,:) = mean(q(range,:));
end
avg_q(T-avg_t:T,1) = mean(q(T-avg_t:T,1));
avg_q(T-avg_t:T,2) = mean(q(T-avg_t:T,2));