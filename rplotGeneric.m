
%filename = ["appliance.rplt"];
filename = ["redrestresult.rplt"];
%filename = [folder sprintf('\\multiresult%d.rplt',(day))]

fileID = fopen(filename,'r');

formatSpec = '%s%[^\n\r]';
dataArray = textscan(fileID, formatSpec, 'Delimiter', '',  'ReturnOnError', false);

fclose(fileID);
fclose('all');

txtdata = dataArray{:,1};
thickness = 2;
figure1 = figure;
h = datacursormode(figure1);
set(h,'UpdateFcn',@dataTipCallback)
axis_count = 0;
new_axis = 0;
range = [];
my_legend = {};
colorpacks
color_pack_id = 1;
color_id = 1;
graph_count = 0;
axis_vars = {};
power_axis = 0;
for i = 1:length(txtdata)
    line = txtdata{i};
    c =  textscan(line,['%s' repmat('%f',1,sum(line==','))],'Delimiter',',');
    entry = c{1};
    data = [c{2:end}];
    if ~isempty(strfind(entry{1},'axis'))
        axis_count = axis_count + 1;
        new_axis = 1;
        range = data;
        axis_label = entry{1}(6:end);
        if ~isempty(strfind(axis_label,'Power'))
            %range(2) = range(2)/1.5;
            power_axis = 1;
        else
            power_axis = 0;
        end
        [s,e] = regexp(axis_label,'\([A-Za-z0-9_ ]*\)')
        unit = axis_label(s:e)
        axis_vars{end+1} = axis_label;
        
    else
         if ~isempty(strfind(entry{1},'quantity'))
             display('Ok')
         end
%          if (strcmp(entry{1}(1:3),'avg') || ~isempty(strfind(entry{1},'power'))) && ~strcmp(entry{1}(1:9),'avg_power')
%              continue
%          end
%          if (strcmp(entry{1}(1:3),'avg')) && ~strcmp(entry{1}(1:9),'avg_power')
         if ~isempty(strfind(entry{1},'House')) & ~isempty(strfind(entry{1},'power'))
             continue
         end
         if ~isempty(strfind(entry{1},'avg_indoor_temperature'))
             thickness = 4;
             manual_color = 1;
             color = [1,0,0];
         else
             thickness = 2;
             manual_color = 0;
         end
         if ~isempty(strfind(entry{1},'cleared_quantity'))
             thickness = 2;
             manual_color = 1;
             color = [1,0,1];
         end
         
        graph_count = graph_count+1;
        my_legend{end+1} = entry{1};
        t = data(1:2:end);

        value = data(2:2:end);
        [t, sortIndices] = sort(t);
        value = value(sortIndices);
%         if ~isempty(strfind(entry{1},'avg_power'))
%             value = value/100;  
%         end
        color_id = mod(graph_count,5)+1;
        pack_id = mod(round(graph_count / 5)+1,5)+1;
        if (manual_color == 0)
            color = color_pack(color_id,:,pack_id)/255;
        end
        if new_axis == 1
            if axis_count == 1
                plot(t,value,'LineWidth',thickness,'ZData',value,'DisplayName',[entry{1} unit])
                
                ylim(range);
                xlim([min(t),max(t)])
                xlabel('Time (hours)')
                hold on;
            else
                axis_count
                addaxis(t,value,range,'LineWidth',thickness,'Color',color,'ZData',value,'DisplayName',[entry{1} unit])
                hold on;
            end
        else
            color_id
            pack_id
            addaxisplot(t,value,axis_count,'LineWidth',thickness,'Color',color,'ZData',value,'DisplayName',[entry{1} unit])
            hold on;
        end
        new_axis = 0;
    end
end


%legend(my_legend)
for i=1:size(axis_vars,2) 
    addaxislabel(i,axis_vars(i));
end

title('Aggregated power profile of 1000 dishwashers');

% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes2,[0 239.999972222]);
% Uncomment the following line to preserve the Y-limits of the axes
% ylim(axes2,[0 100000]);
%box(axes1,'on');
% Create textarrow
annotation(figure1,'textarrow',[0.618093567923949 0.5859375],...
    [0.717616116796679 0.153354632587859],'String',{'loading for washer 1'});

% Create textarrow
annotation(figure1,'textarrow',[0.73348809797608 0.729166666666667],...
    [0.677078729425461 0.15814696485623],'String',{'loading for washer 2'});

% Create textarrow
annotation(figure1,'textarrow',[0.378468368479467 0.321960767402637],...
    [0.657608695652174 0.375063592519483],...
    'String',{'Dishwasher loadshape data from survey'});

% Create textarrow
annotation(figure1,'textarrow',[0.216426193118757 0.199747010119595],...
    [0.747282608695652 0.579312590301377],...
    'String',{'Aggregated Power of dishwashers'});

% Create textarrow
annotation(figure1,'textarrow',[0.32630410654828 0.316315205327414],...
    [0.0560652173913043 0.133152173913043],'String',{'DR event'});
