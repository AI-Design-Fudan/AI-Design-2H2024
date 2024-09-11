import argparse
from PIL import Image
import os

def resize_image(image_path, output_size=(1280, 784)):
    try:
        # 打开图片
        with Image.open(image_path) as img:
            # 调整图片尺寸
            img_resized = img.resize(output_size, Image.LANCZOS)
            
            # 构造保存路径，保存到当前目录
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            new_image_path = f"{name}_resized{ext}"
            
            # 保存新的图片
            img_resized.save(new_image_path)
            print(f"图片已成功保存为: {new_image_path}")
    except Exception as e:
        print(f"处理图片时出错: {e}")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Resize an image to 1280x784.')
    parser.add_argument('--path', type=str, required=True, help='Path to the image to be resized.')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调整图片
    resize_image(args.path)

if __name__ == "__main__":
    main()
