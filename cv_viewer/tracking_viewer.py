import cv2
from cv_viewer import *


def cvt(pt, scale):
    """
    Function that scales point coordinates
    """
    out = [pt[0] * scale[0], pt[1] * scale[1]]
    return out


def render_2D(left_display, img_scale, objects, is_tracking_on):
    """
    Parameters
        left_display (np.array): numpy array containing image data
        img_scale (list[float])
        objects (list[sl.ObjectData])
    """
    overlay = left_display.copy()

    # 呈现骨骼点和骨架
    for obj in objects:
        if render_object(obj, is_tracking_on):
            if len(obj.keypoint_2d) > 0:
                color = generate_color_id_u(obj.id)
                # 绘制骨架连接
                for part in SKELETON_BONES:
                    kp_a = cvt(obj.keypoint_2d[part[0].value], img_scale)
                    kp_b = cvt(obj.keypoint_2d[part[1].value], img_scale)

                    # 确保关键点在画面内
                    if (kp_a[0] < left_display.shape[1] and kp_a[1] < left_display.shape[0]
                            and kp_b[0] < left_display.shape[1] and kp_b[1] < left_display.shape[0]
                            and kp_a[0] > 0 and kp_a[1] > 0 and kp_b[0] > 0 and kp_b[1] > 0):
                        cv2.line(left_display, (int(kp_a[0]), int(kp_a[1])), (int(kp_b[0]), int(kp_b[1])), color, 1,
                                 cv2.LINE_AA)

                # 获取脊柱基点
                left_hip = obj.keypoint_2d[sl.BODY_PARTS.LEFT_HIP.value]
                right_hip = obj.keypoint_2d[sl.BODY_PARTS.RIGHT_HIP.value]
                spine = (left_hip + right_hip) / 2
                kp_spine = cvt(spine, img_scale)
                kp_neck = cvt(obj.keypoint_2d[sl.BODY_PARTS.NECK.value], img_scale)
                # 确保关键点在画面内
                if (kp_spine[0] < left_display.shape[1] and kp_spine[1] < left_display.shape[0]
                        and kp_neck[0] < left_display.shape[1] and kp_neck[1] < left_display.shape[0]
                        and kp_spine[0] > 0 and kp_spine[1] > 0 and kp_neck[0] > 0 and kp_neck[1] > 0
                        and left_hip[0] > 0 and left_hip[1] > 0 and right_hip[0] > 0 and right_hip[1] > 0):
                    cv2.line(left_display, (int(kp_spine[0]), int(kp_spine[1])), (int(kp_neck[0]), int(kp_neck[1])),
                             color, 1, cv2.LINE_AA)

                # 骨骼点
                for kp in obj.keypoint_2d:
                    cv_kp = cvt(kp, img_scale)
                    if (cv_kp[0] < left_display.shape[1] and cv_kp[1] < left_display.shape[0]):
                        cv2.circle(left_display, (int(cv_kp[0]), int(cv_kp[1])), 3, color, -1)
                if (kp_spine[0] < left_display.shape[1] and kp_spine[1] < left_display.shape[0]
                        and left_hip[0] > 0 and left_hip[1] > 0 and right_hip[0] > 0 and right_hip[1] > 0):
                    cv2.circle(left_display, (int(kp_spine[0]), int(kp_spine[1])), 3, color, -1)

    cv2.addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display)
