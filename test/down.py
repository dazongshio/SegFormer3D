import kagglehub

# Specify download directory
download_dir = "/home/niu/Data/Brats/"

# Download latest version to the specified directory
path = kagglehub.dataset_download("dschettler8845/brats-2021-task1", path=download_dir)

print("Path to dataset files:", path)
