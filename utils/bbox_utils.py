# Some calculation for bounding box
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position(bbox, method_id):
    x1, y1, x2, y2 = bbox
    if method_id == 1: 
        return (int((x1 + x2) / 2), int((y1 + y2)*0.52)) # Middle x of bbox, maxium y of bbox
    elif method_id == 2:
        return (int((x1 + x2) / 2), int((y1 + y2)*0.55)) # Middle x of bbox, maxium y of bbox

def get_closest_keypoint_index(point, keypoints, keypoint_candidate):
   closest_distance = float('inf') # Infinity
   key_point_index = keypoint_candidate[0]
   for keypoint_indix in keypoint_candidate:
       keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1] # Get x,y
    #    distance_y = abs(point[1]-keypoint[1]) # Measure y dis
       distance = measure_distance(keypoint, point)
       if distance<closest_distance:
    #    if distance_y<closest_distance:
        #    closest_distance = distance_y
           closest_distance = distance
           key_point_index = keypoint_indix
    
   return key_point_index

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1, p2):
    # return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])
    return p1[0]-p2[0], p1[1]-p2[1]

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))