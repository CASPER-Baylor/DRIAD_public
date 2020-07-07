function plot = Add_Ion_Den_To_2D_Plot(fig, grid_x, grid_z, density)

style = Load_Plot_Styles();
colormap(style.color_cmap);
caxis([0 2]) 
contour_lvls = linspace(0, 2,64); % levels for contour lines
[~, plot] = contourf(fig, grid_z,grid_x, density, contour_lvls, 'Linestyle','none');
