function Animate_Ion_Potential( dir_name, run_name )

%% Get and Load Values

params = Get_Params( dir_name, run_name );

if params.NUM_DUST > 0
    [dust_pos,~,~] = Get_Dust_Trace( dir_name, run_name );
    dust_pos = dust_pos * 1e3; % conver from m to mm 
end 

[grid, ion_potential] = Get_Ion_Potential( dir_name, run_name );

fig_style = Load_Plot_Styles();

%% Add Potential From Outside Ions

grid_x = grid.gridx;
grid_z = grid.gridy;

% potential is -Integral(E dot ds)
pot_outside = -( 0.5 * grid_x.^2 .*(params.P10X+params.P12X*grid_z.^2+params.P14X*grid_z.^4)...
    +0.5*(grid_z.^2.*(params.P01Z+params.P21Z*grid_x.^2))...
    +0.25*grid_z.^4.*(params.P03Z + params.P23Z*grid_x.^2) + (params.P05Z*grid_z.^6)/6);

% convert from m to mm
grid = grid.Set_Grid( grid.Get_GridX() .* 1e-3, grid.Get_GridY() .* 1e-3 ); 

ion_potential =  ion_potential + pot_outside - 3.6;

%% Plot Properties

value_range = [-2, 2];

figure
hold on

axis equal
set(gcf, 'color','w');

%xlim([grid_z(1,1) grid_z(end,end)])
%ylim([grid_x(1,1) grid_x(end,end)])

xlabel('x (mm)')
ylabel('z (mm)')

set( gcf,'position', fig_style.single_figure_pos); 
set( gca, 'position', fig_style.single_axis_pos); 

c = colorbar;
c.Label.String = 'n_i/n_{i0}';
c.Position = fig_style.single_colorbar_pos;
caxis(value_range) 


colormap( fig_style.scale_cmap);
contour_lvls = linspace(value_range(1), value_range(2), 64);


%% Make Video

set(gca, 'NextPlot','replacechildren');

vid = VideoWriter([ dir_name filesep 'potential.avi']);
vid.FrameRate = 15;
open(vid);
hold on

start_time = 0;
end_time = 200;
per_step_size = (end_time - start_time) / ion_potential.Get_Num_Steps();
ion_potential = ion_potential.Set_Times( start_time:per_step_size:(end_time - per_step_size) );

start_time = ion_potential.Get_Time(1);
end_time = ion_potential.Get_Time( ion_potential.Get_Num_Steps() );
step_size = (end_time-start_time)/15;

for time = start_time:step_size:(end_time - step_size) 
     
     ion_plot = grid.Copy_Values( ion_potential.Time_Average( time, time+step_size ) ) ...
                    .Plot_Contourf( gca, contour_lvls );
     
     grid_plot = grid.Plot_Grid( gca );
                 
    if params.NUM_DUST > 0
       error('not implemented')
        %avg_pos = reshape( squeeze(dust_pos(i:i+skip_size-1,:,:)) , [1,3]);
        %dust_plot = Add_Dust_To_2D_Plot( gca, dust_pos(i,:,:), 2, params.RAD_CYL);
    end
    
    title(['Ion Potential        Time = ' num2str(time,'%4.3f')])
    
    frame = getframe(gcf);
    writeVideo(vid,frame)
    
    delete(ion_plot);
    delete(grid_plot);
    if params.NUM_DUST > 0
        delete(dust_plot);
    end
end

close(vid)
close gcf
