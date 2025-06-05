Manual snapshot scan:
1) ros2 launch zed_wrapper zed_camera.launch.py camera_model:=<camera model>
2) ros2 launch zed_operator zed_operator.launch.py
3) ros2 service call /zed_cam_snapshot 