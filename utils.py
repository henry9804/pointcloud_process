import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial.transform import Rotation as R

def pc2_to_numpy(msg):
    """
    Converts a PointCloud2 message to an Nx3 numpy array (x, y, z).
    
    Parameters:
        msg (sensor_msgs.msg.PointCloud2): The PointCloud2 message.
        
    Returns:
        np.array: Nx3 array where each row contains x, y, z (float64).
    """
    # Get raw data buffer from the PointCloud2 message
    dtype_list = {'names': [], 'formats': [], 'offsets': [],
                    'itemsize': msg.point_step}
    for field in msg.fields:
        dtype_list['names'].append(field.name)
        dtype_list['formats'].append(np.uint32 if field.name == 'rgb' else np.float32)
        dtype_list['offsets'].append(field.offset)
    
    dtype = np.dtype(dtype_list)
    data = np.frombuffer(msg.data, dtype=dtype)

    # Extract x, y, z fields
    xyz = np.vstack((data["x"], data["y"], data["z"])).T
    # Extract rgb fields
    if 'rgb' in data.dtype.names:
        rgb_uint32 = data['rgb'].view(np.uint32)
        r = ((rgb_uint32 >> 16) & 255).astype(np.float32) / 255.0
        g = ((rgb_uint32 >> 8)  & 255).astype(np.float32) / 255.0
        b = (rgb_uint32 & 255).astype(np.float32) / 255.0
        rgb = np.vstack((r, g, b)).T
        xyz = np.hstack((xyz, rgb))

    return xyz

def numpy_to_pc2(header, points):
    assert points.shape[-1] == 3, "Input must be [x, y, z]"
    # Flatten the points array
    points = points.reshape(-1, 3).astype(np.float32)

    # Define the fields of the PointCloud2 message
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    # Create the PointCloud2 message
    pc2_msg = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        fields=fields,
        is_bigendian=False,
        point_step=12,  # 4 bytes per field * 3 fields
        row_step=12 * points.shape[0],
        data=points.tobytes(),
        is_dense=True
    )

    return pc2_msg

def numpy_to_pc2_rgb(header, points):
    assert points.shape[-1] == 6, "Input must be [x, y, z, r, g, b]"
    # Flatten the points array
    points = points.reshape(-1, 6).astype(np.float32)

    # unnormalize RGB values if necessary
    if points[:, 3:].max() <= 1.0:
        points[:, 3:] = points[:, 3:] * 255.0

    # pack RGB into single float32 value
    rgb_uint32 = (
        (points[:, 3].astype(np.uint32) << 16) |
        (points[:, 4].astype(np.uint32) << 8) |
        (points[:, 5].astype(np.uint32))
    )
    rgb_float = rgb_uint32.view(np.float32)

    points = np.hstack((points[:, :3], rgb_float.reshape(-1, 1)))

    # Define the fields of the PointCloud2 message
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]

    # Create the PointCloud2 message
    pc2_msg = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        fields=fields,
        is_bigendian=False,
        point_step=16,  # 4 bytes per field * 4 fields
        row_step=16 * points.shape[0],
        data=points.tobytes(),
        is_dense=True
    )

    return pc2_msg

def transform_points(points, T): # points: Nx3, T: TransformStamped
    translation = np.array([T.transform.translation.x, T.transform.translation.y, T.transform.translation.z]).reshape(3, 1)
    orientation = [T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
    rot_mat = R.from_quat(orientation).as_matrix()
    
    transformed = (rot_mat @ points.T) + translation
    return transformed.T

def project(points_3d, K, dist_coeffs):
    # points_3d: Nx3
    rvec = np.zeros((3, 1), dtype=np.float32)  # no rotation
    tvec = np.zeros((3, 1), dtype=np.float32)  # no translation
    
    # OpenCV expects float32
    points_3d = np.ascontiguousarray(points_3d, dtype=np.float32)
    K = np.ascontiguousarray(K, dtype=np.float32)
    dist_coeffs = np.ascontiguousarray(dist_coeffs, dtype=np.float32)

    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)  # Nx2
    # projected_points = projected_points.clip([0, 0], [1279, 719])
    return projected_points

def rgb_to_hsv(rgb):
    """
    Convert RGB values to HSV.
    
    Parameters:
        rgb (np.array): Nx3 array of RGB values.
        
    Returns:
        np.array: Nx3 array of HSV values.
    """
    hsv = np.zeros_like(rgb)
    
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    delta = maxc - minc
    
    hsv[:, 2] = maxc  # Value channel
    
    mask = delta > 0
    hsv[mask, 1] = delta[mask] / maxc[mask]  # Saturation channel
    
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    hsv[mask, 0] = np.where(
        r[mask] == maxc[mask], (g[mask] - b[mask]) / delta[mask],
        np.where(g[mask] == maxc[mask], 2.0 + (b[mask] - r[mask]) / delta[mask], 4.0 + (r[mask] - g[mask]) / delta[mask])
    )
    
    hsv[:, 0] = (hsv[:, 0] / 6.0) % 1.0  # Normalize hue to [0, 1]
    
    return hsv