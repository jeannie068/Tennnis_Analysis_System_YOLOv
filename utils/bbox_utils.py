# Some calculation for bounding box
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))