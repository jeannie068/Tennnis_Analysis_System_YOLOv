from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../') # One folder before
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, people_detections):
        people_detections_first_frame = people_detections[0]
        chosen_player = self.choose_players(court_keypoints, people_detections_first_frame)
        filtered_player_detections = []
        for player_dict in people_detections:
            # 將挑好的player store in filtered_player_dict
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, people_dict):
        distances = []
        for track_id, bbox in people_dict.items():
            people_center = get_center_of_bbox(bbox) # Calculate player's center

            # Find the closest person to court to select two player
            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(people_center, court_keypoint)
                min_distance = min(min_distance, distance)
                # if distance < min_distance:
                #     min_distance = distance
            distances.append((track_id, min_distance))
        
        # Sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks id
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    # Use one frame detact to detact mutiple frames
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        # If 先前已detect
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections
    
    # One frame only
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0] # Multi-Object; Remember the tracks done before
        id_name_dict = results.names

        player_dict = {} # Key: player_id, Output: bounding box
        # Only player, exculde other object
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0] # Bounding box
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

    # Draw Bounding Boxes on output video
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections): # Loop same time
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    