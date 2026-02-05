function Animate_Ion_Density( dir_name, run_name )

%% Get and Load Values

params = Get_Params( dir_name, run_name );

%[dust_pos,~,~] = Get_Dust_Trace( dir_name, run_name );
%dust_pos = dust_pos * 1e3; % conver from m to mm 

[grid, ion_density] = Get_Ion_Density( dir_name, run_name );
% convert from super ion density to real ion density and normalize 
% by the unperturbed plasma density
ion_density = ion_density .* params.SUPER_ION_MULT ./ params.DEN_FAR_PLASMA;

fig_style = Load_Plot_Styles();

%% Plot Properties

figure
hold on

axis equal
set(gcf, 'color','w');

%xlim([grid_z(1,1) grid_z(end,end)])
%ylim([grid_x(1,1) grid_x(end,end)])

ylabel('x (mm)')
xlabel('z (mm)')

set( gcf,'position', fig_style.single_figure_pos); 
set( gca, 'position', fig_style.single_axis_pos); 

c = colorbar;
c.Label.String = 'n_i/n_{i0}';
c.Position = fig_style.single_colorbar_pos;
caxis([0 2]) 

colormap( fig_style.color_cmap);
contour_lvls = linspace(0, 2, 64);


%% Make Video

set(gca, 'NextPlot','replacechildren');

vid = VideoWriter([ dir_name filesep 'density.avi']);
vid.FrameRate = 15;
open(vid);
hold on

start_time = 0;
end_time = 200;
per_step_size = (end_time - start_time) / ion_density.Get_Num_Steps();
ion_density = ion_density.Set_Times( start_time:per_step_size:(end_time - per_step_size) );

start_time = ion_density.Get_Time(1);
end_time = ion_density.Get_Time( ion_density.Get_Num_Steps() );
step_size = (end_time-start_time)/2;

for time = start_time:step_size:(end_time - step_size) 
     
    %density = density.Set_Values( ion_density(:,:,i) );
    %density = density.Set_Values( mean(ion_density(:,:,15:20),3) );
    
    ion_plot = grid.Copy_Values( ion_density.Time_Average( time, time+step_size )' ).Plot_Contourf( gca, contour_lvls );
    
    %avg_pos = reshape( squeeze(dust_pos(i:i+skip_size-1,:,:)) , [1,3]);
    %dust_plot = Add_Dust_To_2D_Plot( gca, dust_pos(i,:,:), 2, params.RAD_CYL);
    
    title(['Ion Density        Time = ' num2str(time,'%4.3f')])
    
    frame = getframe(gcf);
    writeVideo(vid,frame)
    
    delete(ion_plot);
    %delete(dust_plot);
    
end

close(vid)    
close gcf
