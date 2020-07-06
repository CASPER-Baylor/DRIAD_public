function index = Binary_Float_Search( vals, val )

vals = squeeze(vals)';

l_index = 0;
r_index = size(vals,1);
found = false; 

if( vals(end) < val )
   found = true;
   index = size(vals,1);
end

while( ~found )
   
   c_index = ceil( (l_index + r_index) / 2 );
   
   if( vals(c_index) < val )
      l_index = c_index;
   elseif( vals(c_index) > val )
      r_index = c_index;
   else
      index = c_index;
      found = true;
   end
      
   if(r_index - l_index == 1)
      found = true;
      index = l_index;
   elseif r_index < l_index
      error('Algorithm Error')
   end
   
end
