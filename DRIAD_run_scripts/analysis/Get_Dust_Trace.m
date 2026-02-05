function [dust_pos, dust_vel, dust_acc] = Get_Dust_Trace( dir_name, run_name )

params = Get_Params( dir_name, run_name );

% the dust-pos-trace.txt file
data = csvread([dir_name '/' run_name '_dust-pos-trace.txt']); %moving dust
    
increment = params.NUM_DUST + 1;
time = data(1:increment:end,1);
data(1:increment:end,:) = [];
    
dust_pos = zeros(length(time), params.NUM_DUST,3);
dust_acc = zeros(length(time), params.NUM_DUST,3);
dust_vel = zeros(length(time), params.NUM_DUST,3);

for i = 1:length(time)
    range = (1:params.NUM_DUST) + (i-1)*params.NUM_DUST;
    dust_pos(i,:,:) = data(range,1:3);
    dust_vel(i,:,:) = data(range,4:6);
    dust_acc(i,:,:) = data(range,7:9);
end
