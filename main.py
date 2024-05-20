from utils import (read_video, 
                   save_video,
                   measure_distance,
                   convert_pixel_distance_to_meters,
                   draw_player_stats,
                   speed_estimator
                   )
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy


def main():
    # Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True, # False for first time
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True, # False for first time
                                                 stub_path="tracker_stubs/ball_detections.pkl"
                                                 )
    ## Use Pandas to help estimate the missing detect of ball
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    
    # Court Line Detector model
    court_model_path = 'models/keypoints_model.pth'
    # court_model_path = 'models/model_tennis_court_det.pt'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[1]) # Give only one frame
    
    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0]) 
    
    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    
    # Convert player & ball positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                                            ball_detections,
                                                                                                                            court_keypoints)
    # Analysis Board
    player_stats_data_df = speed_estimator(ball_shot_frames, ball_mini_court_detections, player_mini_court_detections, video_frames)
    
    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints(output_video_frames, court_keypoints)
    
    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    
    
    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    save_video(output_video_frames, "output_video/output_video.avi")

if __name__ == "__main__":
    main()