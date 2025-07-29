import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from utils import pc2_to_numpy, numpy_to_pc2_rgb, rgb_to_hsv

class PCColorMask:
    def __init__(self, topic_name):
        rospy.init_node('pc_color_mask', anonymous=True)

        self.bridge = CvBridge()
        img_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        self.hsv = None
        self.clicked = False

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        while not self.clicked:
            cv2.imshow("Image", self.cv_image)
            if cv2.waitKey(10) & 0xFF == 27:  # exit with ESC
                break
        cv2.destroyAllWindows()

        rospy.Subscriber(topic_name, PointCloud2, callback=self.pc_callback)
        
        self.pointcloud_pub = rospy.Publisher('/masked_color_points', PointCloud2, queue_size=1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.cv_image is not None:
            b, g, r = self.cv_image[y, x]
            self.hsv = rgb_to_hsv(np.array([[r, g, b]]) / 255.0)
            rospy.loginfo(f"Clicked pixel (x={x}, y={y}) -> RGB=({r}, {g}, {b}), HSV={self.hsv[0]}")
            self.clicked = True

    def pc_callback(self, msg):
        points = pc2_to_numpy(msg)
        hue_threshold = 0.1
        saturation_threshold = 0.3
        value_threshold = 0.2
        points_hsv = rgb_to_hsv(points[:, 3:])
        mask = (np.abs(points_hsv[:, 0] - self.hsv[0, 0]) < hue_threshold) \
             & (np.abs(points_hsv[:, 1] - self.hsv[0, 1]) < saturation_threshold) \
             & (np.abs(points_hsv[:, 2] - self.hsv[0, 2]) < value_threshold)

        if len(points[mask]) == 0:
            rospy.logwarn("No points match the color mask.")
            return
        header = msg.header
        pcl_msg = numpy_to_pc2_rgb(header, points[mask])
        self.pointcloud_pub.publish(pcl_msg)

if __name__ == '__main__':
    try:
        PCColorMask('/camera/depth/color/points')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass