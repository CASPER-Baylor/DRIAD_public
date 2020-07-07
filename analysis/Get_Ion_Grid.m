function [grid_x, grid_z] = Get_Ion_Grid( dir_name, run_name )

params = Get_Params( dir_name, run_name );

grid_data = csvread([dir_name filesep run_name '_ion-den.txt']);
    
grid_pts = grid_data(1:params.NUM_GRID_PTS,:);
grid_x = reshape(grid_pts(:,1),params.RESX,params.RESZ);
grid_z = reshape(grid_pts(:,3),params.RESX,params.RESZ);