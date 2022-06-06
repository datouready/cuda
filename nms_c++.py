vector<Box> cpu_decode(float* predict, int rows, int cols, float confidence_threshold = 0.25f, float nms_threshold = 0.45f){
    
    vector<Box> boxes;#存放需要的bbox信息
    int num_classes = cols - 5;
    for(int i = 0; i < rows; ++i){
        float* pitem = predict + i * cols;#获取当前bbox的指针数组
        float objness = pitem[4]；#获得confidence
        if(objness < confidence_threshold)
            continue;

        float* pclass = pitem + 5;#获得类别指针
        int label     = std::max_element(pclass, pclass + num_classes) - pclass；#获得标签索引
        float prob    = pclass[label];
        float confidence = prob * objness;
        if(confidence < confidence_threshold)
            continue;

        float cx     = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];
        float left   = cx - width * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width * 0.5;
        float bottom = cy + height * 0.5;
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }

    std::sort(boxes.begin(), boxes.end(), [](Box& a, Box& b){return a.confidence > b.confidence;});
    std::vector<bool> remove_flags(boxes.size());
    std::vector<Box> box_result;
    box_result.reserve(boxes.size());

    auto iou = [](const Box& a, const Box& b){
        float cross_left   = std::max(a.left, b.left);
        float cross_top    = std::max(a.top, b.top);
        float cross_right  = std::min(a.right, b.right);
        float cross_bottom = std::min(a.bottom, b.bottom);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) 
                         + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < boxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = boxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < boxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = boxes[j];
            if(ibox.label == jbox.label){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    return box_result;
}