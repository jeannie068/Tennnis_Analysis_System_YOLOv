import numpy as np
import cv2
import sys 
sys.path.append('../')
from utils import (measure_distance,
                   convert_pixel_distance_to_meters
                   )
import constants
from mini_court import MiniCourt
import pandas as pd
from copy import deepcopy

def speed_estimator(ball_shot_frames, ball_mini_court_detections, player_mini_court_detections, video_frames):
    player_stats_data = [{
        'frame_num': 0,
        'player_1_total_player_distance': 0,
        'player_1_total_player_times': 0,
        'player_1_last_player_speed': 0,

        'player_2_total_player_distance': 0,
        'player_2_total_player_times': 0,
        'player_2_last_player_speed': 0,
    } ]

    ball_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
    } ]
    
    for frame_i in range(len(video_frames[::3])-1):
        start_frame = frame_i*3 # Previous
        end_frame = frame_i*3+3 # Currennt
        player_move_time_in_seconds = (end_frame-start_frame)/24 # 24fps
        # Get distance covered by the player 1
        distance_covered_by_player_pixels = measure_distance(player_mini_court_detections[start_frame][1],
                                                             player_mini_court_detections[end_frame][1])
        distance_covered_by_player_meters = convert_pixel_distance_to_meters(distance_covered_by_player_pixels,
                                                                             constants.DOUBLE_LINE_WIDTH,
                                                                             MiniCourt(video_frames[0]).get_width_of_mini_court()
                                                                             )
        speed_of_player = distance_covered_by_player_meters/player_move_time_in_seconds * 3.6

        # Player 1 analysis board
        current_ball_stats = deepcopy(player_stats_data[-1]) # Previous 
        current_ball_stats['frame_num'] = end_frame
        current_ball_stats['player_1_total_player_distance'] += distance_covered_by_player_meters
        current_ball_stats['player_1_total_player_times'] += player_move_time_in_seconds
        current_ball_stats['player_1_last_player_speed'] = speed_of_player

        # Get distance covered by the player 2
        distance_covered_by_player_pixels = measure_distance(player_mini_court_detections[start_frame][2],
                                                             player_mini_court_detections[end_frame][2])
        distance_covered_by_player_meters = convert_pixel_distance_to_meters(distance_covered_by_player_pixels,
                                                                             constants.DOUBLE_LINE_WIDTH,
                                                                             MiniCourt(video_frames[0]).get_width_of_mini_court()
                                                                             )
        speed_of_player = distance_covered_by_player_meters/player_move_time_in_seconds * 3.6

        # Player 2 analysis board
        current_ball_stats['player_2_total_player_distance'] += distance_covered_by_player_meters
        current_ball_stats['player_2_total_player_times'] += player_move_time_in_seconds
        current_ball_stats['player_2_last_player_speed'] = speed_of_player

        player_stats_data.append(current_ball_stats) # Store back

    # Ball
    for ball_shot_ind in range(len(ball_shot_frames)):
        start_frame = ball_shot_frames[ball_shot_ind] # Currennt
        end_frame = start_frame+20 # Next
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           MiniCourt(video_frames[0]).get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6 * 1.5

        # Player who the ball
        player_positions = player_mini_court_detections[start_frame]
        player_id_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                    ball_mini_court_detections[start_frame][1]))

        # Analysis board
        current_ball_stats = deepcopy(ball_stats_data[-1]) # Previous 
        current_ball_stats['frame_num'] = start_frame
        current_ball_stats[f'player_{player_id_shot_ball}_number_of_shots'] += 1
        current_ball_stats[f'player_{player_id_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_ball_stats[f'player_{player_id_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        ball_stats_data.append(current_ball_stats) # Store back

    ball_stats_data_df = pd.DataFrame(ball_stats_data) # Change to pandas datafram (easy to display)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    ball_stats_data_df = pd.merge(frames_df, ball_stats_data_df, on='frame_num', how='left') # Merge
    # print(player_stats_data_df.to_markdown)
    ball_stats_data_df = ball_stats_data_df.ffill()

    player_stats_data_df = pd.DataFrame(player_stats_data) # Change to pandas datafram (easy to display)
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left') # Merge
    player_stats_data_df = player_stats_data_df.ffill()
    player_stats_data_df = pd.merge(player_stats_data_df, ball_stats_data_df, on='frame_num', how='left') # Merge
    # print(player_stats_data_df.to_markdown)

    # Calculate average speed
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_distance']/player_stats_data_df['player_1_total_player_times'] * 3.6
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_distance']/player_stats_data_df['player_2_total_player_times'] * 3.6
    
    return player_stats_data_df

def draw_player_stats(output_video_frames, player_stats):

    for index, row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width=350
        height=230

        start_x = frame.shape[1]-380
        start_y = frame.shape[0]-500
        end_x = start_x+width
        end_y = start_y+height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        text = "     Player 1     Player 2"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+80, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Shot Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        
        text = "avg. S. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. P. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output_video_frames
