import cv2

def read_frames_from_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frames = []
    if capture.isOpened():
        while True:
            ret, img = capture.read()
            if not ret:
                break
            frames.append(img)
    else:
        print('视频打开失败！')
    return frames

def create_collage(frames, num_segments=60, num_output_images=5):
    segment_length = len(frames) // num_segments
    output_folder = 'output_images'  # 指定一个文件夹名
    for i in range(num_output_images):
        collage = frames[i * segment_length: (i + 1) * segment_length]
        output_image_path = f'{output_folder}/output_image_{i}.jpg'  # 添加文件名到文件夹路径
        cv2.imwrite(output_image_path, collage)
        print(f'Saved {output_image_path}')

if __name__ == "__main__":
    video_path = 0
    frames = read_frames_from_video(video_path)
    create_collage(frames)


