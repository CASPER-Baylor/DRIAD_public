classdef LinearTimeSeries
   
   properties
      values
      time
      indices
      shape
      set
      time_dim
   end
   
   methods
      
      function obj = LinearTimeSeries()
         obj.set = [false, false];
      end
      
      function obj = Set_Times( obj, new_times )
         if( obj.set(2) == true && size( obj.values, obj.time_dim) ~= length(new_times) )
            error( 'The time and value dimensions do not match')
         end
         
         obj.set(1) = true;
         obj.time = new_times;
         
      end
      
      function obj = Set_Values( obj, new_values, new_time_dim )
         
         obj.values = new_values;
         
         obj.shape = size(new_values);
         obj.shape(new_time_dim) = [];
         
         found = false;
         for i = length(size( obj.values )):-1:1
            if( size( obj.values, i ) > 1 && ~found )
               max_dim = i;
               found = true;
            end
         end
         
         purmutation = 1:max_dim;
         purmutation(max_dim) = new_time_dim;
         purmutation(new_time_dim) = max_dim;
         
         if( length(purmutation) == 1 )
            purmutation(2) = 2;
         end
         
         obj.values = permute(obj.values, purmutation);
         obj.time_dim = max_dim;
         
         per_step_index = 1;
         
         if obj.time_dim > 1
            for i = 1:obj.time_dim - 1
               per_step_index = per_step_index .* size(obj.values, i);
            end
         end
         
         tmp_indices = zeros(size(obj.values,obj.time_dim), per_step_index);
         
         tmp_indices(:,1) = 1 + ((1:size(obj.values,obj.time_dim)) - 1) * per_step_index;
         
         if per_step_index > 1
            for i = 2:per_step_index
               tmp_indices(:,i) = tmp_indices(:,1) + i - 1;
            end
         end
         
         obj.indices = tmp_indices;
         
         
         if( obj.set(1) == true && size( obj.values, obj.time_dim) ~= length(obj.time) )
            error( 'The time and value dimensions do not match')
         end
         
         obj.set(2) = true;
         
      end
  
      function obj = plus( obj, in )
         if( isa( in, 'LinearTimeSeries') )
            for i = 1:numel(obj.values)
               obj.values(i) = obj.values(i) + in.values(i);
            end
         else
            for i = 1:numel(obj.values)
               obj.values(i) = obj.values(i) + in;
            end
         end
      end
      
      function obj = minus( obj, in )
         if( isa( in, 'LinearTimeSeries') )
            for i = 1:numel(obj.values)
               obj.values(i) = obj.values(i) - in.values(i);
            end
         else
            for i = 1:numel(obj.values)
               obj.values(i) = obj.values(i) - in;
            end
         end
      end
      
      function obj = times( obj, in )
         if( isa( in, 'LinearTimeSeries') )
            for i = 1:numel(obj.values)
               obj.values(i) = obj.values(i) .* in.values(i);
            end
         else
            for i = 1:numel(obj.values)
               obj.values(i) = obj.values(i) .* in;
            end
         end
      end
      
      function obj = rdivide( obj, in )
         if( isa( in, 'LinearTimeSeries') )
            for i = 1:numel(obj.values)
               obj.values(i) = obj.values(i) ./ in.values(i);
            end
         else
            for i = 1:numel(obj.values)
               obj.values(i) = obj.values(i) ./ in;
            end
         end
      end
      
      function obj = uminus( obj )
         for i = 1:numel(obj.values)
            obj.values = -obj.values;
         end
      end
      
      function val = Get_Num_Steps(obj)
         if obj.set(1) == true
            val = length( obj.time );
         elseif obj.set(2) == true
            val = size( obj.values, obj.time_dim );
         else
            error('The values or time have not been set')
         end
      end
      
      function val = Get_Step(obj, step)
         if(obj.set(2) == false)
            error('The values have not been set')
         end
         if( ~sum( size(obj.shape) ~= size(1) ) )
            val = obj.values( obj.indices(step,:) );
         else
            val = reshape( obj.values( obj.indices(step,:) ), obj.shape );
         end
      end
      
      function val = Get_Time( obj, step )
         if( obj.set(1) == false )
            error('The time have not been set')
         end
         val = obj.time(step);
      end
      
      function val = Index_Average( obj, begining, ending )
         if(obj.set(2) == false)
            error('The values have not been set')
         end
         
         num = ending - begining + 1;
         val = obj.Get_Step(begining) ./ num;
         for i = begining+1:ending
            val = val +  obj.Get_Step(i) ./ num;
         end
      end
      
      function val = Time_Average( obj, begining, ending )
         if( obj.set(1) == false )
            error('The time have not been set')
         end
         
         if(obj.set(2) == false)
            error('The values have not been set')
         end
         
         if( ending < obj.time(1) )
            val = obj.Get_Step(1);
         elseif begining > obj.time(end)
            val = obj.Get_Step( length(obj.time) );
         else
            total_time = ending - begining;
            current_time = begining;
            current_step = Binary_Float_Search( obj.time, begining );
            if( current_step < 1 )
               current_step = 1;
            end
            
            val = obj.Get_Step(1) .* 0;
            
            while(current_time < ending && current_step < length(obj.time) )
               
               if( ending > obj.time(current_step + 1) )
                  step_time = obj.time(current_step + 1) - current_time;
                  current_time = current_time + step_time;
               else
                  step_time = ending - current_time;
                  current_time = ending;
               end
               
               val = val + obj.Get_Step(current_step) .* step_time ./ total_time;
               current_step = current_step + 1;
               
            end
            
            if( ending > current_time )
               val = val + ( ending - current_time ) .* obj.Get_Step(length(obj.time)) ./ total_time;
            end
            
         end
         
      end
      
   end
end