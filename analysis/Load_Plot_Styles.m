function params = Load_Plot_Styles()

params.scale_cmap = brewermap(64, 'PuRd');
params.color_cmap = brewermap(64,'RdYlBu');%colormap(jet);
params.gray_cmap = gray(32);
params.single_figure_pos = [350  350 750 295];
params.single_axis_pos = [.088 .103 .775 .815];
params.single_colorbar_pos = [.8859 .1767 .038 .6642];