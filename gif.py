from PIL import Image
import os


def create_gif(image_folder, gif_name, duration=10):
    images = []

    # 找到指定文件夾中的所有JPG文件，按文件名排序
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    image_files.sort()
    count=len(image_files)

    for filename in range(count):
        file_path = os.path.join(image_folder, str(filename)+".jpg")
        images.append(Image.open(file_path))

    # 設定GIF的持續時間，這裡的單位是毫秒
    gif_path = os.path.join(image_folder, gif_name)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


if __name__ == "__main__":
    image_folder = "./gif"  # 替換成實際的圖片文件夾路徑
    gif_name = "output2.gif"
    duration = 20  # 每張圖片顯示的時間（毫秒）

    create_gif(image_folder, gif_name, duration)
