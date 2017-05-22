line_width = 1.5;
marker_size = 7;
xy_font_size = 14;
legend_font_size = 12;
title_font_size = xy_font_size;
for j=1:length(hbits)
    figure;hold on;grid on;
    bit = hbits(j)*hbits(j);
    title([dataset,'@',num2str(bit),'bits']);
    for i= [1:5,7:8]
        p = plot(recall{j,i}, precision{j,i});
        set(p,'Color', gen_color(i));
        set(p,'Marker', gen_marker(i));
        set(p,'LineWidth', line_width);
        set(p,'MarkerSize', marker_size);
    end
    xlabel('Recall');
    ylabel('Precision');
    legend(method{[1:5,7:8]});
    box on;hold off;
end