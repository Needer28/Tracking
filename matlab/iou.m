function score = iou(bb1, bb2)
    % calculates the intersection-over-union of two bounding boxes
    % format of bb: [x_left, y_top, width, height]
    % returns a fraction as the iou score
    
    % turns the bb format into [x_left, y_top, x_right, y_bottom]
    bb1 = [bb1(1:2), bb1(1) + bb1(3), bb1(2) + bb1(4)];
    bb2 = [bb2(1:2), bb2(1) + bb2(3), bb2(2) + bb2(4)];
    
    % calculates the intersection
    inter_x_left   = max([bb1(1), bb2(1)]);
    inter_y_top    = max([bb1(2), bb2(2)]);
    inter_x_right  = min([bb1(3), bb2(3)]);
    inter_y_bottom = min([bb1(4), bb2(4)]);
    
    % checks if the iou is 0
    if inter_x_left >= inter_x_right || inter_y_top >= inter_y_bottom
        score = 0;
        return;
    end
    
    % calculates the iou score
    inter_area = (inter_x_right - inter_x_left) *...
                 (inter_y_bottom - inter_y_top);
    area_1 = (bb1(3) - bb1(1)) * (bb1(4) - bb1(2));
    area_2 = (bb2(3) - bb2(1)) * (bb2(4) - bb2(2));
    union_area = area_1 + area_2 - inter_area;
    score = inter_area / union_area;
end