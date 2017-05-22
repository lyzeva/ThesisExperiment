line_width = 1.5;
marker_size = 3;
xy_font_size = 14;
legend_font_size = 12;
title_font_size = xy_font_size;
for j=1:length(assess.hbits)
    figure;hold on;grid on;
    bit = assess.hbits(j);
    title([assess.dataset,'@',num2str(bit),'bits']);
    for i= 1:size(assess.method,2)
        p = plot(recall_kNN{j,i}, precision_kNN{j,i});
        set(p,'Color', gen_color(i));
        set(p,'Marker', gen_marker(i));
        set(p,'LineWidth', line_width);
        set(p,'MarkerSize', marker_size);
    end
    xlabel('Recall');
    ylabel('Precision');
    legend(assess.method);
    box on;hold off;
end