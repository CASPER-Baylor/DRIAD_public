% Reads in all the parameters from the <run_name>_params.txt
% 
% Parameters:
%   dir_name: the path to the directory containing the parameter file. 
%       It should NOT be terminated with a file seperato.
%       Correct: /path/to/directory
%       Incorrect: /path/to/directory/
%   run_name: The run name prepended to all IonWake output file names.
%
% Return:
%   params: a structure containing all of the parameters in the 
%       parameter file.
%
function params = Get_Params( dir_name, run_name )

fid = fopen( [ dir_name filesep run_name '_params.txt' ] );

tline = fgetl(fid);
while (tline ~= -1)
    [~,q] = find(tline == '%');
    tempval = deblank(tline(1:q(1)-1));    
    tempvar = deblank(tline(q(end)+1:end));    
    eval(['params.' tempvar ' = ' tempval ';']);
    tline = fgetl(fid);
end

        