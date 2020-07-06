classdef Scaler2DGrid
   
   properties
      gridx
      gridy
      values
      set % set(1) = values and set(2) = grid
   end
   
   methods
      
      function obj = Scaler2DGrid()
         obj.set = [false, false];
      end
      
      function obj = Set_Grid( obj, new_gridx, new_gridy )
         
         if( numel(new_gridx) ~= size(new_gridx, 1) )
            error(['The new_gridx parameter must be 1D'...
               ' but was of size (' num2str(size(new_gridx)) ')'] )
         end
         
         if( numel(new_gridy) ~= size(new_gridy, 1) )
            error(['The new_gridy parameter must be 1D'...
               ' but was of size (' num2str(size(new_gridy)) ')'] )
         end
         
         if( obj.set(1) && length(new_gridx) ~= size(obj.values, 1) )
            error(['The new_gridx parameter must be the same'...
               'length as the contained values 1st dimension' ])
         end
         
         if( obj.set(1) && length(new_gridy) ~= size(obj.values, 2) )
            error(['The new_gridy parameter must be the same'...
               'length as the contained values 2nd dimension' ])
         end
         
         ny = length( new_gridy );
         nx = length( new_gridx );
         
         obj.gridy = repmat( reshape( new_gridy, [1, ny]), nx, 1 );
         obj.gridx = repmat( reshape( new_gridx, [nx, 1]), 1, ny );
         
         obj.set(2) = true;
         
      end
      
      function obj = Set_Values( obj, values )
         
         if ( numel(values) ~= size(values, 1) * size(values, 2))
            error('The values parameter must be 2D')
         end
         
         if (obj.set(2) && size( values, 1 ) ~= size(obj.gridx,1))
            error(['The 1st dimension of the values parameter must'...
               ' be the same length as the x grid'])
         end
         
         if (obj.set(2) && size( values, 2) ~= size(obj.gridy,2))
            error(['The 2nd dimension of the values parameter must'...
               ' be the same length as the y grid'])
         end
         
         obj.values = values;
         obj.set(1) = true;
         
      end
      
      function val = Get_GridX( obj )
         val = obj.gridx(:,1);
      end
      
      function val = Get_GridY( obj )
         val = obj.gridy(1,:)';
      end      
      
      function obj = plus( obj, in )
         if( isa( in, 'Scaler2DGrid') )
            obj.values = obj.values + in.values;
         else
            obj.values = obj.values + in;
         end
      end
      
      function obj = minus( obj, in )
         if( isa( in, 'Scaler2DGrid') )
            obj.values = obj.values - in.values;
         else
            obj.values = obj.values - in;
         end
      end
      
      function obj = times( obj, in )
         if( isa( in, 'Scaler2DGrid') )
            obj.values = obj.values .* in.values;
         else
            obj.values = obj.values .* in;
         end
      end
      
      function obj = rdivide( obj, in )
         if( isa( in, 'Scaler2DGrid') )
            obj.values = obj.values ./ in.values;
         else
            obj.values = obj.values ./ in;
         end
      end
      
      function obj = uminus( obj )
         obj.values = -obj.values;
      end
      
      function plot = Plot_Contourf( obj, fig, contour_lvls )
         
         if( ~obj.set(1) || ~obj.set(2) )
            error('The grid and values must be set befor plotting')
         end
         
         [~, plot] = contourf(fig,obj.gridx, obj.gridy, obj.values, contour_lvls, 'Linestyle','none');
         
      end
      
      function plot = Plot_Grid( obj, fig )
         plot = scatter( fig, reshape( obj.gridx, [],1), reshape(obj.gridy, [],1), '.b' );
      end
      
      function obj = Copy_Values( obj, other )
         obj = obj.Set_Values( other.values );
      end
      
      function obj = Copy_Grid( obj, other )
         obj.gridx = other.gridx;
         obj.gridy = other.gridy;
      end
      
      function obj = Clear( obj )
         obj.gridx = [];
         obj.gridy = [];
         obj.values = [];
         obj.set = false;
      end
      
   end
end
