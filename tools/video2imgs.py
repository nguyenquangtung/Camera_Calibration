import cv2


def extract_frames(video_path, output_folder, interval_seconds):
    # Mở video
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Không thể mở tệp video.")
        return

    # Tạo thư mục đầu ra nếu chưa tồn tại
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Đếm frame và thời gian
    frame_count = 0
    elapsed_time = 0

    # Lặp qua các frame
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Lưu frame sau mỗi khoảng thời gian interval_seconds
        if elapsed_time >= interval_seconds:
            output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            frame_count += 1
            elapsed_time = 0
        else:
            elapsed_time += 1 / video_capture.get(cv2.CAP_PROP_FPS)

    # Giải phóng tài nguyên
    video_capture.release()


if __name__ == "__main__":
    # Thực thi hàm
    video_path = r"image\output_cam\1713239415171219.mp4"
    output_folder = r"image"
    interval_seconds = 1  # Khoảng cách giữa các frame là bao nhiêu giây
    extract_frames(video_path, output_folder, interval_seconds)
