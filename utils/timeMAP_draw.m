line_width = 2;
marker_size = 8;
xy_font_size = 14;
legend_font_size = 12;
linewidth = 1.6;
title_font_size = xy_font_size;

for loop = 1:length(assess.loop)
figure; hold on;
for i =  1:size(assess.method,2)
	time = zeros(1,length(assess.hbits));
    MAP = [];
	for j= 1:length(assess.hbits)
		time(j) = res.coding_time{loop}{j,i};
        MAP = [MAP, res.map{loop}{j, i}];
	end
    p = plot(time*1000, MAP);
    set(p,'Color', gen_color(i));
    set(p,'Marker', gen_marker(i));
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end
xlabel('Indexing Time (ms)');

	ylabel('mAP');
	title(assess.dataset);
	legend(assess.method);
	box on; grid on;hold off;
end