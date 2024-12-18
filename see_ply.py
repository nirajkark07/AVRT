import open3d as o3d
import argparse

def visualize_ply(file_path):
    # Load the point cloud from the .ply file
    pcd = o3d.io.read_point_cloud(file_path)
    
    if pcd.is_empty():
        print("Error: The point cloud is empty.")
        return

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a .ply file.")
    parser.add_argument("file_path", type=str, help="Path to the .ply file.")
    args = parser.parse_args()
    
    visualize_ply(args.file_path)