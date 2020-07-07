function [grid, potential] = Get_Ion_Potential( dir_name, run_name )

params = Get_Params( dir_name, run_name );
grid_data = csvread([dir_name filesep run_name '_ion-den.txt']);

grid = Scaler2DGrid;
grid = grid.Set_Grid( grid_data(1:params.RESX:params.NUM_GRID_PTS,3), grid_data(1:params.RESX,1));

tmp_grid_data = grid_data( (params.NUM_GRID_PTS+1):end,2);
grid_data = zeros( ceil(length(tmp_grid_data)/params.NUM_GRID_PTS), params.RESZ, params.RESX );

for( i = 0:size(grid_data,1)-1 )
   for( j = 0:size(grid_data,2)-1 )
      for( k = 0:size(grid_data,3)-1 )
         index = i*params.NUM_GRID_PTS + params.RESX * j + k + 1;
         grid_data(i+1,j+1,k+1) = tmp_grid_data(index);
      end
   end
end

for i = 1:size(grid_data,1)
   tmp = Scaler2DGrid;
   tmp = tmp.Set_Values( squeeze( grid_data(i,:,:) ) );
   potential_array(i,1) = tmp;
end

potential = LinearTimeSeries;
potential = potential.Set_Values(potential_array, 1);




