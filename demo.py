import argparse

import cv2
import numpy as np
import torch
import time
import random

# 加载苹果图片
apple_img = cv2.imread('material/apple.png', cv2.IMREAD_UNCHANGED)
golden_apple_img = cv2.imread('material/golden.png', cv2.IMREAD_UNCHANGED)

# 调整图片大小
apple_img = cv2.resize(apple_img, (50, 50))
golden_apple_img = cv2.resize(golden_apple_img, (50, 50))

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cpu()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


###################

# 生成随机数量的小圆
def generate_food():
    num_points = random.randint(8, 20)
    points = [(random.randint(25, 935), random.randint(25, 695)) for _ in range(num_points)]  # 普通苹果
    golden_apple = None
    if random.random() < 0.3:  # 30%的概率生成金苹果
        golden_apple = (random.randint(25, 935), random.randint(25, 695))
    return points, golden_apple

####################



def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cpu()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1

    # 初始化参数
    running = False  # 游戏是否在运行
    score = 0  # 分数
    start_time = None  # 游戏开始时间
    food_points = []  # 初始化为空，避免重复生成
    countdown_time = 60  # 倒计时时间（秒）
    golden_apple = None  # 初始化金苹果变量

    for img in image_provider:
        # 对捕获的图像进行水平翻转
        img = cv2.flip(img, 1)  # 参数 1 表示水平翻转

        # 调整图像大小（例如放大 1.5 倍）
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        # 更新 orig_img，使其与 img 的大小一致
        orig_img = img.copy()

        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

            key_pose = pose.keypoints[0]  # 鼻子的关键点

            if key_pose[0] != -1 and key_pose[1] != -1:  # 确保关键点有效
                cv2.circle(img, (int(key_pose[0]), int(key_pose[1])), 10, (255, 0, 0), -1)  # 绘制蓝色小球

        # 游戏逻辑，仅在游戏运行时执行
        if running:
            remaining_food = []
            golden_eaten = False  # 标记金苹果是否被吃掉
            for (fx, fy) in food_points:
                food_eaten = False
                for pose in current_poses:
                    nose = pose.keypoints[0]  # 用鼻子作为检测点
                    if nose[0] != -1 and nose[1] != -1:
                        dist_sq = (fx - nose[0])**2 + (fy - nose[1])**2
                        if dist_sq < 600:  # 若距离小于阈值，认为食物被吃掉
                            score += 1   # 普通苹果得分
                            food_eaten = True
                            break
                if not food_eaten:
                    remaining_food.append((fx, fy))
                else:
                    print(f"Food eaten at: ({fx}, {fy})")  # 调试信息
            food_points = remaining_food

            # 检测金苹果是否被吃掉
            if golden_apple:
                for pose in current_poses:
                    nose = pose.keypoints[0]
                    if nose[0] != -1 and nose[1] != -1:
                        dist_sq = (golden_apple[0] - nose[0])**2 + (golden_apple[1] - nose[1])**2
                        if dist_sq < 600:  # 若距离小于阈值，认为金苹果被吃掉
                            score += 5  # 金苹果得分更高
                            golden_eaten = True
                            print(f"Golden apple eaten at: {golden_apple}")  # 调试信息
                            break
                if golden_eaten:
                    golden_apple = None  # 移除金苹果

            # 如果所有食物都被吃掉，则生成新食物
            if not food_points and not golden_apple:
                print("All food eaten, generating new food...")
                food_points, golden_apple = generate_food()

        #----- 绘制食物（苹果图片） -----
        for (fx, fy) in food_points:
            xp = int(fx - 25)
            yp = int(fy - 25)
            if xp >= 0 and yp >= 0 and xp+50 <= img.shape[1] and yp+50 <= img.shape[0]:
                alpha = apple_img[:, :, 3] / 255.0
                for c in range(3):
                    img[yp:yp+50, xp:xp+50, c] = (1.0 - alpha) * img[yp:yp+50, xp:xp+50, c] + alpha * apple_img[:, :, c]
            else:
                print(f"Food point out of bounds: ({fx}, {fy})")  # 调试信息

        #----- 绘制金苹果 -----
        if golden_apple:
            gx, gy = golden_apple
            xg = int(gx - 25)
            yg = int(gy - 25)
            if xg >= 0 and yg >= 0 and xg+50 <= img.shape[1] and yg+50 <= img.shape[0]:
                alpha = golden_apple_img[:, :, 3] / 255.0
                for c in range(3):
                    img[yg:yg+50, xg:xg+50, c] = (1.0 - alpha) * img[yg:yg+50, xg:xg+50, c] + alpha * golden_apple_img[:, :, c]
            else:
                print(f"Golden apple point out of bounds: ({gx}, {gy})")

        # 显示分数和倒计时
        cv2.putText(img, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 显示分数
        cv2.putText(img, f'runtime:{running}', (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if running:
            remaining_time = max(0, countdown_time - int(time.time() - start_time))  # 计算剩余时间
            cv2.putText(img, f'Time Left: {remaining_time}s', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 显示倒计时

        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1
        elif key == ord('a'):  # 按下'a'键开始游戏
            running = True
            start_time = time.time()
            score = 0
            food_points, golden_apple = generate_food()  # 游戏开始时生成食物
        elif key == ord('d'):  # 按下'd'键停止游戏
            running = False

        if running and (time.time() - start_time) >= countdown_time:  # 游戏运行倒计时结束后停止
            running = False
            print(f'Final Score: {score}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
