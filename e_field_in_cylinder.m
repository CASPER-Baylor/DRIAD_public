%e_field_in_cylinder.m
%numerical calculation of electric inside cylinder
%Calculate the potential inside a cylinder of uniformly distributed ions.
%Numerically integrate over the volume.  Subtract this potential from the
%constant potential of uniformly distributed ions to get the potential in a
%cavity.  Then calculate the electric field inside the cavity.  Fit the
%electric field lines with parabolas in (r,z) to determine an analytic 
%solution for the electric field.  
%lsm November 18, 2017

clear
close all

% constants
ion_density = 6e14; %number density per cubic m
TEMP_ELC = 51060; % 1 eV = 29011 K
qe = -1.6022e-19;  % Charge of electron (C)
epsilon_0 = 8.854e-12; %  Permittivity of Free Space (F/m)
kb = 1.38044e-23; %  Boltzmann Constant (J/K)
qi = -qe;
DEBYE = sqrt(epsilon_0*kb*TEMP_ELC/(ion_density*qe*qe)); % --- Electron Debye length (m)
%RAD_SIM = 2*DEBYE;
%% Set up a grid for calculating potential
% This is for a single slice in the xz plane

cyl_rad = 1.5*DEBYE;
cyl_height = 6*DEBYE; %half the cylinder height as limit in z is -ht:+ht

resx = 20; %resolution of grid spacing
resz = 50;  
 bin_edgesx = linspace(-cyl_rad, cyl_rad, resx+1);
 bin_edgesz = linspace(-cyl_height, cyl_height, resz+1);
dx = (bin_edgesx(2)-bin_edgesx(1))/DEBYE; %in units of debye length
dz = (bin_edgesz(2)-bin_edgesz(1))/DEBYE; %in units of debye length
 bin_centersx = diff(bin_edgesx)/2+bin_edgesx(1:end-1);
 bin_centersz = diff(bin_edgesz)/2+bin_edgesz(1:end-1);
 
 [X,Z] = meshgrid(bin_centersx, bin_centersz);
num_grid_pts = resx*resz;
grid_pts = [reshape(X,num_grid_pts,1), zeros(num_grid_pts,1),reshape(Z,num_grid_pts,1),];
%% 3D grid for positions of ions inside the cylinder
nx = resx*1.5; %use a different resolution so that points don't overlap grid
nz = resz*1.5;
ddx = (2*cyl_rad/nx);
ddz = (2*cyl_height/nz);
ionx = -cyl_rad:ddx:cyl_rad;
iony = ionx;
ionz = -cyl_height:ddz:cyl_height;
[ix, iy, iz] = meshgrid(ionx, iony, ionz);

%% potential for ions inside a cylinder
num_ion_pts = (nx+1)*(nx+1)*(nz+1);
 V_ionsIn_cyl = zeros(num_grid_pts,1);
ionPos = [reshape(ix,num_ion_pts,1),reshape(iy,num_ion_pts,1),reshape(iz,num_ion_pts,1),];
ion_rad = ionPos(:,1).^2+ionPos(:,2).^2;
[a,~] = find(ion_rad <= cyl_rad^2 );
ions_in_cyl = ionPos(a,:);
%%
% need distance between each of the ions and grid points
% since there are a lot of ions, break them up into smaller batches
num_batches = 4;
ions_per_batch = floor(length(a)/num_batches);
POS1=repmat(reshape(grid_pts,[num_grid_pts,1,3]),[1,ions_per_batch,1]);
q_in_box = qi* ion_density * ddx*ddx*ddz;
for batch = 1:num_batches
    start_index = (batch-1)*ions_per_batch+1;
    end_index = batch*ions_per_batch;
    batch_ions = ions_in_cyl(start_index:end_index,:);
    %each row of pos2 is a repeat of the first one.
    POS2=repmat(reshape(batch_ions,[1,ions_per_batch,3]),[num_grid_pts,1,1]);
    %find the distance between every grid pt and all ions in batch
    %dist_pts_inv is num_grid_pts x ions_per_batch in size
    dist_pts=sqrt(sum((POS1-POS2).^2,3)); 
    dum = ((q_in_box/(4*pi*epsilon_0)).*exp(-dist_pts/DEBYE))./dist_pts;
    V_ionsIn_cyl = V_ionsIn_cyl + sum(dum,2);
    disp(batch)
end

figure(1)
contour(X/DEBYE,Z/DEBYE,reshape(V_ionsIn_cyl-2.5, resz, resx));
axis equal
title('V_{In_cyl}')

%% Calculate the electric field
V_cavity = reshape(V_ionsIn_cyl - 2.5, resz,resx);
figure
[C,h] = contour(X/DEBYE,Z/DEBYE,reshape(V_cavity, resz, resx));
axis equal
clabel(C,h)
title('Potential inside cylindrical cavity')
[Ex, Ez] = gradient(V_cavity, dx*DEBYE, dz*DEBYE);

% Need the range of Ex and Ez to specify contour level lines
maxEx = max(max(Ex));
divx = maxEx/4;
maxEz = max(max(Ez));
divz = maxEz/5;

levelsx = -maxEx:divx:maxEx;
levelsz = -maxEz:divz:maxEz;
figure
subplot(1,2,1)
[Cx,hx] =contour(X/DEBYE,Z/DEBYE,Ex,levelsx,'Showtext','on');
title('Horizontal electric field')
subplot(1,2,2)
[Cz,hz] =contour(X/DEBYE,Z/DEBYE,Ez,levelsz,'Showtext','on');
title('Vertical electric field')

%% Now get fits to the surfaces
x = reshape(X,num_grid_pts,1);
y = reshape(Z, num_grid_pts,1);
zEx = reshape(Ex,num_grid_pts,1);
zEz = reshape(Ez,num_grid_pts,1);

 fEx = fit( [x,y], zEx, 'poly15' );
 coeffx = coeffvalues(fEx);
 fEz = fit([x,y],zEz,'poly25');
 coeffz = coeffvalues(fEz);
 format shortg
 disp('Fit for Ex ')
 disp('      p10x     p12xz^2      p14xz^4')
 disp(coeffx([2,6,10]))
 disp('Fit for Ez ')
 disp('      p01z     p21x^2z      p03z^3        p23x^2z^3      p05z^5')
 disp(coeffz([3,7,9,13,15]))

%  %% Compare magnitude of terms
%  %for Ex
%  total = feval(fEx,[cyl_rad,cyl_height]);
%  p00 = coeffx(1)/total*100
%  p10 = coeffx(2)*cyl_rad /total*100
%  p01 = coeffx(3)*cyl_height/total*100
%  p11 = coeffx(4)*cyl_rad*cyl_height/total*100
%  p02 = coeffx(5)*cyl_height^2/total*100
%  p12 = coeffx(6)*cyl_rad*cyl_height^2/total*100
%  p03 = coeffx(7)*cyl_height^3/total*100
%  p13 = coeffx(8)*cyl_rad*cyl_height^3/total*100
%  p04 = coeffx(9)*cyl_height^4/total*100
%  p14 = coeffx(10)*cyl_rad*cyl_height^4/total*100
%  p05 = coeffx(11)*cyl_height^5
%  disp(' ')
%   %for Ez
%  total = feval(fEz,[cyl_rad,cyl_height]);
%  p00 = coeffz(1)/total*100
%  p10 = coeffz(2)*cyl_rad /total*100
%  p01 = coeffz(3)*cyl_height/total*100
%  p20 = coeffz(4)*cyl_rad^2/total*100
%  p11 = coeffz(5)*cyl_rad*cyl_height/total*100
%  p02 = coeffz(6)*cyl_height^2/total*100
%  p21 = coeffz(7)*cyl_rad^2*cyl_height/total*100
%  p12 = coeffz(8)*cyl_rad*cyl_height^2/total*100
%  p03 = coeffz(9)*cyl_height^3/total*100
%  p22 = coeffz(10)*cyl_rad^2*cyl_height^2/total*100
%  p13 = coeffz(11)*cyl_rad*cyl_height^3/total*100
%  p04 = coeffz(12)*cyl_height^4/total*100
%  p23 = coeffz(13)*cyl_rad^2*cyl_height^3/total*100
%  p14 = coeffz(14)*cyl_rad*cyl_height^4/total*100
%  p05 = coeffz(15)*cyl_height^5/total*100
