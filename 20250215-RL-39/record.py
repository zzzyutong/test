import os
import shutil
from config_loader import get_file_name

def create_directory(path):
    """创建目录，如果目录已存在则忽略"""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"目录已创建或已存在: {path}")
    except Exception as e:
        print(f"创建目录失败: {path}. 错误: {e}")

def copy_files(file_list, src_dir, dest_dir):
    """复制指定的文件列表从源目录到目标目录"""
    for file_name in file_list:
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
            print(f"已复制: {src_file} 到 {dest_file}")
        else:
            print(f"源文件不存在: {src_file}")

def copy_all_csv(src_dir, dest_dir):
    """复制源目录中的所有CSV文件到目标目录"""
    try:
        files = [f for f in os.listdir(src_dir) if f.lower().endswith('.csv')]
        for file_name in files:
            src_file = os.path.join(src_dir, file_name)
            dest_file = os.path.join(dest_dir, file_name)
            shutil.copy(src_file, dest_file)
            print(f"已复制: {src_file} 到 {dest_file}")
    except FileNotFoundError:
        print(f"源目录不存在或无法访问: {src_dir}")

def main():
    # 1. 定义filename
    filename = get_file_name()  # 请替换为您需要的名称

    current_dir = os.getcwd()

    # 2. 创建当前目录下的filename文件夹并复制.svg文件
    svg_files = ['angle.svg', 'branch.svg', 'price.svg', 'tao.svg']
    dest_svg_dir = os.path.join(current_dir, filename)
    create_directory(dest_svg_dir)
    copy_files(svg_files, current_dir, dest_svg_dir)

    # 3. 在dataset文件夹下创建filename文件夹并复制15个CSV文件
    dataset_files = [
        'data_to_test.csv', 'data_to_test_original.csv', 'data_to_test_with_headers.csv',
        'test_fake_data.csv', 'test_fake_data_original.csv', 'test_fake_data_with_headers.csv',
        'test1_fake_data.csv', 'test1_fake_data_original.csv', 'test1_fake_data_with_headers.csv',
        'test2_fake_data.csv', 'test2_fake_data_original.csv', 'test2_fake_data_with_headers.csv',
        'test3_fake_data.csv', 'test3_fake_data_original.csv', 'test3_fake_data_with_headers.csv'
    ]
    src_dataset_dir = os.path.join(current_dir, 'dataset')
    dest_dataset_dir = os.path.join(src_dataset_dir, filename)
    create_directory(dest_dataset_dir)
    copy_files(dataset_files, src_dataset_dir, dest_dataset_dir)

    # 4. 在measurements文件夹下创建filename文件夹并复制所有CSV文件
    src_measurements_dir = os.path.join(current_dir, 'measurements')
    dest_measurements_dir = os.path.join(src_measurements_dir, filename)
    create_directory(dest_measurements_dir)
    copy_all_csv(src_measurements_dir, dest_measurements_dir)

    # 5. 在metric文件夹下创建filename文件夹并复制所有CSV文件
    src_metric_dir = os.path.join(current_dir, 'metric')
    dest_metric_dir = os.path.join(src_metric_dir, filename)
    create_directory(dest_metric_dir)
    copy_all_csv(src_metric_dir, dest_metric_dir)

    # 6. 在power_flow文件夹下创建filename文件夹并复制所有CSV文件
    src_power_flow_dir = os.path.join(current_dir, 'power_flow')
    dest_power_flow_dir = os.path.join(src_power_flow_dir, filename)
    create_directory(dest_power_flow_dir)
    copy_all_csv(src_power_flow_dir, dest_power_flow_dir)

    print("所有操作已完成。")

if __name__ == "__main__":
    main()
