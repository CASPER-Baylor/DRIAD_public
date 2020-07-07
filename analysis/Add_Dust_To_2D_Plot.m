function plot = Add_Dust_To_2D_Plot( fig, dust_pos, index, scale)

style = Load_Plot_Styles();

cmap = style.gray_cmap;
num_color = length(cmap);

dust_pos = reshape(squeeze(dust_pos),[1,3]);
color = round(abs(dust_pos(:,index))/(0.55*scale) * num_color )+1;
color(color>num_color) = num_color;

plot = scatter( ...
    fig, ... The figure
    dust_pos(:,3), ... the dust x position
    dust_pos(:,1), ... the dust y position
    40, ... the marker size
    'MarkerEdgeColor', [1 1 1], ... the marker edge color (white)
    'MarkerFaceColor', cmap(color,:) ... the marker face color
    );
