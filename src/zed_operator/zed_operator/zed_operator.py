import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime
import traceback
import time
import struct
import open3d as o3d
from std_srvs.srv import SetBool
import pickle
from arm_interfaces.srv import TimedCloud


class ZedOperaterNode(Node):
    def __init__(self):
        super().__init__('zed_operator_node')
        
        depth_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.serv_cb_group = MutuallyExclusiveCallbackGroup()
        
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, depth_qos)
        self.fused_cloud = self.create_subscription(PointCloud2, '/zed/zed_node/mapping/fused_cloud', self.fused_cloud_callback, depth_qos)
        # self.colour_image = self.create_subscription(Image, '/zed/zed_node/right/image_rect_color', self.colour_image_callback, depth_qos)
        # self.point_cloud = self.create_subscription(PointCloud2, '/zed/zed_node/point_cloud/cloud_registered', self.point_cloud_callback, depth_qos)
        
        self.get_logger().info('1')
        self.fused_cloud_client = self.create_client(SetBool, '/zed/zed_node/enable_mapping')
        req = SetBool.Request()
        req.data = False
        self.fused_cloud_client.call_async(req)
        self.get_logger().info("Made mapping toggle serv and turned off")

        self.bridge = CvBridge()
        
        self.want_depth_image = False
        self.latest_depth_image = None
        self.want_fused_cloud = False
        self.latest_fused_point_cloud = None

        self.fused_point_cloud_timed_service()

        self.get_logger().info("Zed operator node initialised")

    
    def depth_callback(self, msg):
        if not self.want_depth_image:
            return
        self.latest_depth_image = msg
        self.want_depth_image = False

    def fused_cloud_callback(self, msg):
        if not self.want_fused_cloud:
            return
        self.latest_fused_point_cloud = msg

    def fused_point_cloud_timed_service(self):
        def fused_point_cloud_timed(request, response):
            self.want_fused_cloud = True
            req = SetBool.Request()
            req.data = True
            self.fused_cloud_client.call_async(req)
            self.get_logger().info("Started mapping")

            time.sleep(request.duration)

            while not self.latest_fused_point_cloud:
               print("Still don't have cloud??")
            response.fused_cloud = self.latest_fused_point_cloud
            self.want_fused_cloud = False
            
            req.data = False
            self.fused_cloud_client.call_async(req)

            self.get_logger().info("Sent fused cloud")
            self.save_cloud_as_o3d(self.latest_fused_point_cloud)
            return response
        return self.create_service(TimedCloud, '/fused_cloud_over_time', fused_point_cloud_timed, callback_group=self.serv_cb_group)


    def save_cloud_as_o3d(self, cloud_msg):
        pc_data = pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))
        points = []
        colors = []

        for p in pc_data:
            points.append([p[0], p[1], p[2]])

            if len(p) >= 4:
                rgb = p[3]
                s = struct.pack('>f', rgb)
                i = struct.unpack('>l', s)[0]
                r = (i >> 16) & 0x0000ff
                g = (i >> 8) & 0x0000ff
                b = i & 0x0000ff
                colors.append([r/255.0, g/255.0, b/255.0])
            else:
                colors.append([0.5, 0.5, 0.5])

        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(np.array(points))
        
        if colors:
            o3d_pc.colors = o3d.utility.Vector3dVector(np.array(colors))
            
        pc_path = '/local/zed_outs/fused_cloud_example.ply'
        o3d.io.write_point_cloud(pc_path, o3d_pc)
        return


def main(args=None):
  rclpy.init(args=args)

  try:
    node = None
    node = ZedOperaterNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

  except Exception as exception:  
    traceback_logger_node = Node('python_traceback_logger')
    traceback_logger_node.get_logger().error(traceback.format_exc())  
    raise exception

  finally:
    if node is not None:
      node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
  main()