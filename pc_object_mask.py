import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, Image
import numpy as np
import matplotlib.pyplot as plt
import PIL
import message_filters

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration

from utils import pc2_to_numpy, numpy_to_pc2_rgb, transform_points, project

K = np.array([[942.347,   0.0  , 638.685],
                [  0.0  , 942.925, 367.797],
                [  0.0  ,   0.0  ,   1.0  ]])
dist_coeffs = np.array([0.2304, -0.6688, 0.003387, -0.003307, 0.63585])

def refine_mask(masks):
    masks = masks.squeeze().cpu().float()
    masks = masks.permute(1, 2, 0)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).numpy() # OR operation
    return masks

class ObjectDetectionMasking:
    def __init__(self, pointcloud_name, image_name, visualize=False):
        # load model
        self.device = "cuda"

        gdino_id = "IDEA-Research/grounding-dino-tiny"
        self.gdino_processor = AutoProcessor.from_pretrained(gdino_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(self.device)

        sam_id = "facebook/sam-vit-base"
        self.sam_processor = AutoProcessor.from_pretrained(sam_id)
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(sam_id).to(self.device)

        rospy.init_node('pc_to_rgb', anonymous=True)

        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        self.T_depth_to_rgb = tf_buffer.lookup_transform('d435i_color_optical_frame', 'd435i_depth_optical_frame', rospy.Time(0), rospy.Duration(1.0))

        img_sub = message_filters.Subscriber(image_name, Image)
        pc_sub = message_filters.Subscriber(pointcloud_name, PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 1, 0.01)
        ts.registerCallback(self.sync_callback)
        
        self.pointcloud_pub = rospy.Publisher('/object_points', PointCloud2, queue_size=1)

        self.image = None
        self.points_rgb_frame = None
        self.header = None
        self.processing = False
        self.visualize = visualize
        self.num_result = 0

    def sync_callback(self, img_msg, pc_msg):
        if self.processing:
            return
        self.image = PIL.Image.frombuffer("RGB", (img_msg.width, img_msg.height), img_msg.data)
        points = pc2_to_numpy(pc_msg)
        self.points_rgb_frame = transform_points(points[:,:3], self.T_depth_to_rgb)
        self.points_rgb_frame = np.hstack((self.points_rgb_frame, points[:,3:]))
        self.header = pc_msg.header

    def grounded_segmentation(self, text_label):
        if self.image is None:
            return
        self.processing = True

        inputs = self.gdino_processor(images=self.image, text=text_label, return_tensors="pt").to(self.device)        
        with torch.no_grad():
            outputs = self.detector(**inputs)

        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.4,
            target_sizes=[self.image.size[::-1]]
        )
        result = results[0]
        
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            box = box.tolist()
            if("cup" in labels):
                print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")

                if self.visualize:
                    uvs = project(self.points_rgb_frame[:,:3], K, dist_coeffs)
                    uvs = np.round(uvs).astype(np.int32)
                    min = np.min(uvs, axis=0)
                    max = np.max(uvs, axis=0)
                    uvs -= min
                    projected_img = np.zeros((max[1]-min[1]+1, max[0]-min[0]+1, 3), dtype=np.float32)
                    for uv, rgb in zip(uvs, self.points_rgb_frame[:,3:]):
                        projected_img[uv[1], uv[0]] = rgb
                    fig, axes = plt.subplots(2, 1)
                    axes[0].imshow(self.image)
                    axes[0].add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="red", linewidth=2))
                    axes[1].imshow(projected_img)
                    axes[1].axhline(-min[1])
                    axes[1].axhline(720-min[1])
                    axes[1].axvline(-min[0])
                    axes[1].axvline(1280-min[0])
                    axes[1].add_patch(plt.Rectangle((box[0]-min[0], box[1]-min[1]), box[2] - box[0], box[3] - box[1], fill=False, color="red", linewidth=2))
                    plt.savefig(f'plots/result_{self.num_result}.png')
                    self.num_result += 1
                
                inputs = self.sam_processor(images = self.image, input_boxes=[[[box]]], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.segmentator(**inputs)
                
                masks = self.sam_processor.post_process_masks(
                    masks=outputs.pred_masks,
                    original_sizes=inputs.original_sizes,
                    reshaped_input_sizes=inputs.reshaped_input_sizes
                )
                segmentation_mask = refine_mask(masks[0])

                points_uv = project(self.points_rgb_frame[:,:3], K, dist_coeffs)
                points_uv = points_uv.astype(np.int32)
                points_uv = points_uv.clip([0, 0], [1279, 719])
                mask = segmentation_mask[points_uv[:,1], points_uv[:,0]]
                # mask = (points_uv[:,0] > box[0]) & (points_uv[:,0] < box[2]) \
                #      & (points_uv[:,1] > box[1]) & (points_uv[:,1] < box[3])

                header = self.header
                header.frame_id = "d435i_color_optical_frame"
                object_points = self.points_rgb_frame[mask]
                pcl_msg = numpy_to_pc2_rgb(header, object_points)
                self.pointcloud_pub.publish(pcl_msg)
        
        self.processing = False

if __name__ == '__main__':
    processor = ObjectDetectionMasking('/camera/depth/color/points', '/camera/color/image_raw')
    while not rospy.is_shutdown():
        processor.grounded_segmentation("pick up the cup.")
        rospy.sleep(0.1)