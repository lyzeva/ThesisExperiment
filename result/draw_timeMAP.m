line_width = 2;
marker_size = 8;
xy_font_size = 14;
legend_font_size = 12;
linewidth = 1.6;
title_font_size = xy_font_size;

figure; hold on;
for i = 1:size(method,2)
	time = zeros(1,length(hbits));
    MAP = [];
	for j= 1:length(hbits)
		time(j) = coding_time{j,i};
        MAP = [MAP, map{j, i}];
	end
    p = plot(time*1000, MAP);
    set(p,'Color', gen_color(i));
    set(p,'Marker', gen_marker(i));
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end
	xlabel('Indexing Time (ms)');
	ylabel('mAP');
	title(dataset);
	legend(method);
	box on; hold off;
