line_width = 1.5;
marker_size = 3;
xy_font_size = 14;
legend_font_size = 12;
title_font_size = xy_font_size;
for loop = 1:assess.loop
for j=1:length(assess.hbits)
    figure;hold on;grid on;
    title(dataset);
    bit = assess.hbits(j);
    for i = 1:length(assess.method);
        p = plot(1:length(res.precision{loop}{j,i}), res.precision{loop}{j,i}); 
        set(p,'Color', gen_color(i));
        set(p,'Marker', gen_marker(i));
        set(p,'LineWidth', line_width);
        set(p,'MarkerSize', marker_size);
    end
    xlabel('Hamming Distance Threshold');
    ylabel(['Recall','@',num2str(bit),'bits']);
    legend(assess.method);
    box on;hold off;
end
end