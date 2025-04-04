import numpy as np
from geometry_msgs.msg import TransformStamped
import tf_transformations

from typing import Any

def tf_to_se3(transform: TransformStamped.transform) -> np.ndarray:
    """
    Convert a TransformStamped message to a 4x4 SE3 matrix
    """
    q = transform.rotation
    q = [q.x, q.y, q.z, q.w]
    t = transform.translation
    mat = tf_transformations.quaternion_matrix(q)
    mat[0, 3] = t.x
    mat[1, 3] = t.y
    mat[2, 3] = t.z
    return mat


def se3_to_tf(mat: np.ndarray, time: Any, parent: str, child: str) -> TransformStamped:
    """
    Convert a 4x4 SE3 matrix to a TransformStamped message
    """
    obj = TransformStamped()

    # current time
    obj.header.stamp = time.to_msg()

    # frame names
    obj.header.frame_id = parent
    obj.child_frame_id = child

    # translation component
    obj.transform.translation.x = mat[0, 3]
    obj.transform.translation.y = mat[1, 3]
    obj.transform.translation.z = mat[2, 3]

    # rotation (quaternion)
    q = tf_transformations.quaternion_from_matrix(mat)
    obj.transform.rotation.x = q[0]
    obj.transform.rotation.y = q[1]
    obj.transform.rotation.z = q[2]
    obj.transform.rotation.w = q[3]

    return obj
