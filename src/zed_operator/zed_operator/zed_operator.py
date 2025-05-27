import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_srvs.srv import Trigger
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
from std_msgs.msg import Bool
import pickle
import open3d as o3d
from arm_interfaces.srv import TimedCloud
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time
import onnxruntime as ort
from zed_operator import inference


class ZedOperaterNode(Node):
    def __init__(self):
        super().__init__('zed_operator_node')
        
        depth_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.serv_cb_group = MutuallyExclusiveCallbackGroup()
        
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, depth_qos)
        self.fused_cloud = self.create_subscription(PointCloud2, '/zed/zed_node/mapping/fused_cloud', self.fused_cloud_callback, depth_qos)
        self.colour_image = self.create_subscription(Image, '/zed/zed_node/right/image_rect_color', self.colour_image_callback, depth_qos)
        self.point_cloud = self.create_subscription(PointCloud2, '/zed/zed_node/point_cloud/cloud_registered', self.point_cloud_callback, depth_qos)

        self.make_point_cloud = self.create_subscription(Bool, '/make_point_cloud', self.make_point_cloud_callback, depth_qos)
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/foundation_stereo_cloud', depth_qos)

        self.right_sub = self.create_subscription(
            Image,
            '/zed/zed_node/right/image_rect_color',
            self.image_right_rectified_callback,
            qos_profile=depth_qos
        )

        self.left_sub = self.create_subscription(
            Image,
            '/zed/zed_node/left/image_rect_color',
            self.image_left_rectified_callback,
            qos_profile=depth_qos
        )
        
        self.get_logger().info('1')
        self.fused_cloud_client = self.create_client(SetBool, '/zed/zed_node/enable_mapping')
        req = SetBool.Request()
        req.data = False
        self.fused_cloud_client.call_async(req)
        self.get_logger().info("Made mapping toggle serv and turned off")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.save_dir = '/home/canterbury/zed_out/'
        
        self.want_depth_image = False
        self.latest_depth_image = None
        self.want_fused_cloud = False
        self.latest_fused_point_cloud = None
        self.want_colour_image = None
        self.latest_colour_image = None
        self.want_point_cloud = None
        self.latest_point_cloud = None
        self.transform_dict = {}

        self.want_right_img = False
        self.want_left_img = False
        self.save_side_imgs = True
        self.left_img = None
        self.right_img = None
        self.timestamp = None
        self.this_save_dir = None

        self.fused_point_cloud_timed_service()
        self.zed_snapshot_service()

        # self.get_logger().info("starting onnx session")
        # self.ort_session = ort.InferenceSession('/home/canterbury/stereo_model/foundation_stereo_960.onnx',
        #                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.get_logger().info("Zed operator node initialised")

    
    # Zed cam topic subscribers
    def depth_callback(self, msg):
        if not self.want_depth_image:
            return
        self.latest_depth_image = msg
        self.want_depth_image = False

    def colour_image_callback(self, msg):
        if not self.want_colour_image:
            return
        self.latest_colour_image = msg
        self.want_colour_image = False

    def point_cloud_callback(self, msg):
        if not self.want_point_cloud:
            return
        self.latest_point_cloud = msg
        self.want_point_cloud = False

    def fused_cloud_callback(self, msg):
        if not self.want_fused_cloud:
            return
        self.latest_fused_point_cloud = msg


    # get FoundationStereo inferenced point cloud
    def make_point_cloud_callback(self, msg):
        self.want_left_img = True
        self.want_right_img = True

        # do inference with foundation stereo model
        inference.onnx_inference(self.ort_session)

        # convert to PointCloud2 and publish with time stamp


    # save zed data at time of service call
    def zed_snapshot_service(self):
        def zed_snapshot(request, response):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            this_save_dir = self.save_dir + timestamp + '/'
            os.makedirs(this_save_dir, exist_ok=True)
            self.timestamp = timestamp
            self.this_save_dir = this_save_dir
            # try:
            self.want_colour_image = True
            self.want_depth_image = True
            self.want_point_cloud = True
            self.want_fused_cloud = True
            self.want_right_img = True
            self.want_left_img = True

            # get enu to zed cam transform
            now = Time()
            trans = self.tf_buffer.lookup_transform(
            target_frame='zcam_link',
            source_frame='local_enu',
            # source_frame='base_link',
            time=now
            )
            self.get_logger().info(
            f"Got transform at time {trans.header.stamp.sec}.{trans.header.stamp.nanosec:09d}: "
            f"translation = ({trans.transform.translation.x:.3f}, "
            f"{trans.transform.translation.y:.3f}, "
            f"{trans.transform.translation.z:.3f}), "
            f"rotation (quat) = ({trans.transform.rotation.x:.3f}, "
            f"{trans.transform.rotation.y:.3f}, "
            f"{trans.transform.rotation.z:.3f}, "
            f"{trans.transform.rotation.w:.3f})"
            )
            self.transform_dict[timestamp] = trans
            with open(os.path.join(self.save_dir, 'transforms.pkl'), 'wb') as f:
                pickle.dump(self.transform_dict, f)

            # save locally
            while not self.latest_depth_image:
                pass
            cv_depth_path = os.path.join(this_save_dir, f"depth_cv_{timestamp}.npy")
            cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='32FC1')
            np.save(cv_depth_path, cv_depth)

            raw_depth_path = os.path.join(this_save_dir, f"depth_raw_{timestamp}.pkl")
            with open(raw_depth_path, 'wb') as f:
                pickle.dump(self.latest_depth_image, f)

            while not self.latest_colour_image:
                pass
            color_path = os.path.join(this_save_dir, f"colour_{timestamp}.png")
            col_img = self.bridge.imgmsg_to_cv2(self.latest_colour_image, desired_encoding='bgr8')
            cv2.imwrite(color_path, col_img)
            
            while not self.latest_point_cloud:
                pass
            pc_path = os.path.join(this_save_dir, f"pointcloud_o3d_{timestamp}.ply")
            o3d_pc = self.save_cloud_as_o3d(self.latest_point_cloud)
            o3d.io.write_point_cloud(pc_path, o3d_pc)

            raw_pc_path = os.path.join(this_save_dir, f"pointcloud_raw_{timestamp}.pkl")
            with open(raw_pc_path, 'wb') as f:
                pickle.dump(self.latest_point_cloud, f)

            response.success = True
            self.get_logger().info(f"Saved snapshot to {this_save_dir}")
            # except:
                # response.success = False
                # self.get_logger().info("Failed")

            self.latest_colour_image = None
            self.latest_depth_image = None
            self.latest_point_cloud = None
            return response
        return self.create_service(Trigger, '/zed_cam_snapshot', zed_snapshot, callback_group=self.serv_cb_group)
    

    # capture a fused point cloud over a specified time period
    def fused_point_cloud_timed_service(self):
        def fused_point_cloud_timed(request, response):
            self.want_fused_cloud = True
            req = SetBool.Request()
            req.data = True
            self.fused_cloud_client.call_async(req)
            self.get_logger().info("Started mapping")

            time.sleep(request.duration)

            while not self.latest_fused_point_cloud:
               self.get_logger().info("Still don't have cloud??")
            response.fused_cloud = self.latest_fused_point_cloud
            self.want_fused_cloud = False
            
            req.data = False
            self.fused_cloud_client.call_async(req)

            self.get_logger().info("Sent fused cloud")
            self.save_cloud_as_o3d(self.latest_fused_point_cloud)
            return response
        return self.create_service(TimedCloud, '/fused_cloud_over_time', fused_point_cloud_timed, callback_group=self.serv_cb_group)
    

    def image_right_rectified_callback(self, msg: Image):
        if not self.want_right_img:
            return
        
        self.get_logger().info(
            f"Right Rectified image received from ZED\tSize: {msg.width}x{msg.height} - "
            f"Ts: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} sec"
        )
       
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.right_img = cv_image

        if self.save_side_imgs:
            # ts_sec = msg.header.stamp.sec
            # ts_nsec = msg.header.stamp.nanosec
            filename = f"right_{self.timestamp}.png"
            filepath = os.path.join(self.this_save_dir, filename)

            success = cv2.imwrite(filepath, cv_image)
            if success:
                self.get_logger().info(f"Saved right image: {filepath}")
            else:
                self.get_logger().error(f"Failed to save right image to {filepath}")
        self.want_right_img = False


    def image_left_rectified_callback(self, msg: Image):
        if not self.want_left_img:
            return
        
        self.get_logger().info(
            f"Left  Rectified image received from ZED\tSize: {msg.width}x{msg.height} - "
            f"Ts: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} sec"
        )

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.left_img = cv_image

        if self.save_side_imgs:
            # ts_sec = msg.header.stamp.sec
            # ts_nsec = msg.header.stamp.nanosec
            filename = f"left_{self.timestamp}.png"
            filepath = os.path.join(self.this_save_dir, filename)

            success = cv2.imwrite(filepath, cv_image)
            if success:
                self.get_logger().info(f"Saved left image: {filepath}")
            else:
                self.get_logger().error(f"Failed to save left image to {filepath}")
        self.want_left_img = False


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
            
        # pc_path = '/local/zed_outs/fused_cloud_example.ply'
        # o3d.io.write_point_cloud(pc_path, o3d_pc)
        return o3d_pc


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