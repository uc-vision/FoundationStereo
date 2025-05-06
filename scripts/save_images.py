import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime
import threading
import time
import struct
import open3d as o3d
import pickle

class DepthSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        
        self.save_dir = os.path.join(os.getcwd(), 'depth_images')
        os.makedirs(self.save_dir, exist_ok=True)
        
        depth_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, depth_qos)

        self.colour_image = self.create_subscription(Image, '/zed/zed_node/right/image_rect_color', self.colour_image_callback, depth_qos)

        self.point_cloud = self.create_subscription(PointCloud2, '/zed/zed_node/point_cloud/cloud_registered', self.point_cloud_callback, depth_qos)
        
        self.bridge = CvBridge()
        
        self.latest_depth_image = None
        self.image_lock = threading.Lock()

        self.latest_colour_image = None
        self.latest_point_cloud = None
        
        save_inc = 5.0
        self.timer = self.create_timer(save_inc, self.save_images)
        
        self.get_logger().info('Saving images every 5 seconds to: ' + self.save_dir)
    
    def depth_callback(self, msg):
        try:

            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            with self.image_lock:
                self.latest_depth_image = {
                    'raw': cv_depth,
                    'header': msg.header
                }
                
            u = msg.width // 2
            v = msg.height // 2
            center_depth = cv_depth[v, u]
            self.get_logger().info(f"Center distance: {center_depth:.3f} m")
            
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def colour_image_callback(self, msg):
        self.latest_colour_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def point_cloud_callback(self, msg):
        self.latest_point_cloud = msg

    
    def save_images(self):
        with self.image_lock:
            if self.latest_depth_image is None:
                self.get_logger().warn("No depth image received yet")
                return
            
            depth_data = self.latest_depth_image.copy()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            raw_depth_path = os.path.join(self.save_dir, f"depth_raw_{timestamp}.npy")
            np.save(raw_depth_path, depth_data['raw'])
            
            
            self.get_logger().info(f"Saved depth image to {raw_depth_path}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving depth image: {e}")

        if self.latest_colour_image is not None:
            try:
                color_path = os.path.join(self.save_dir, f"color_{timestamp}.png")
                cv2.imwrite(color_path, self.latest_colour_image)
                
                self.get_logger().info(f"Saved color image to {color_path}")
                
            except Exception as e:
                self.get_logger().error(f"Error saving color image: {e}")

        try:
            o3d_pc = self.convert_pointcloud2_to_o3d(self.latest_point_cloud)
            
            pc_path = os.path.join(self.save_dir, f"pointcloud_{timestamp}.ply")
            o3d.io.write_point_cloud(pc_path, o3d_pc)
            
            raw_pc_path = os.path.join(self.save_dir, f"pointcloud_raw_{timestamp}.msg")
            with open(raw_pc_path, 'wb') as f:
                pickle.dump(self.latest_point_cloud, f)
            
            self.get_logger().info(f"Saved point cloud to {pc_path}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving point cloud: {e}")


    def convert_pointcloud2_to_o3d(self, cloud_msg):
        """
        Convert a ROS PointCloud2 message to an Open3D point cloud
        """
        # Read points from the PointCloud2 message
        pc_data = pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))
        points = []
        colors = []

        for p in pc_data:
            # Extract XYZ coordinates
            points.append([p[0], p[1], p[2]])
            
            # Extract RGB color
            # RGB is packed into a float in the 'rgb' field
            if len(p) >= 4:  # Check if RGB data exists
                rgb = p[3]
                # Convert RGB from float to bytes
                s = struct.pack('>f', rgb)
                i = struct.unpack('>l', s)[0]
                # Extract colors (0-255)
                r = (i >> 16) & 0x0000ff
                g = (i >> 8) & 0x0000ff
                b = i & 0x0000ff
                # Normalize to 0-1 for Open3D
                colors.append([r/255.0, g/255.0, b/255.0])
            else:
                # Default color if no RGB data
                colors.append([0.5, 0.5, 0.5])

        # Create Open3D point cloud
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(np.array(points))
        
        if colors:
            o3d_pc.colors = o3d.utility.Vector3dVector(np.array(colors))
            
        return o3d_pc


def main(args=None):
    rclpy.init(args=args)
    
    depth_saver = DepthSaverNode()
    
    try:
        rclpy.spin(depth_saver)
    except KeyboardInterrupt:
        pass
    finally:
        depth_saver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()