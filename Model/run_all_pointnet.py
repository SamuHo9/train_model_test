import os
import subprocess

tasks = [
    {
        "name": "Ds004469_Left",
        "train": "Model/Ds004469/left/ds004469_left_Train_augmented_xyz_coords.csv",
        "test": "Model/Ds004469/left/ds004469_left_coef_features_test_xyz_coords.csv",
        "out": "Model/Ds004469/left/PointNet_Results"
    },
    {
        "name": "Ds004469_Right",
        "train": "Model/Ds004469/right/ds004469_right_Train_augmented_xyz_coords.csv",
        "test": "Model/Ds004469/right/ds004469_right_coef_features_test_xyz_coords.csv",
        "out": "Model/Ds004469/right/PointNet_Results"
    },
    {
        "name": "Ds005602_Left",
        "train": "Model/Ds005602/left/ds005602_left_Train_augmented_xyz_coords.csv",
        "test": "Model/Ds005602/left/ds005602_left_coef_features_test_xyz_coords.csv",
        "out": "Model/Ds005602/left/PointNet_Results"
    },
    {
        "name": "Ds005602_Right",
        "train": "Model/Ds005602/right/ds005602_right_Train_augmented_xyz_coords.csv",
        "test": "Model/Ds005602/right/ds005602_right_coef_features_test_xyz_coords.csv",
        "out": "Model/Ds005602/right/PointNet_Results"
    },
    {
        "name": "Ds004469Train_Ds005602test_Left",
        "train": "Model/Ds004469Train_Ds005602test/left/ds004469_left_full_augmented_xyz_coords.csv",
        "test": "Model/Ds004469Train_Ds005602test/left/ds005602_left_coef_features_xyz_coords.csv",
        "out": "Model/Ds004469Train_Ds005602test/left/PointNet_Results"
    },
    {
        "name": "Ds004469Train_Ds005602test_Right",
        "train": "Model/Ds004469Train_Ds005602test/right/ds004469_right_full_augmented_xyz_coords.csv",
        "test": "Model/Ds004469Train_Ds005602test/right/ds005602_right_coef_features_xyz_coords.csv",
        "out": "Model/Ds004469Train_Ds005602test/right/PointNet_Results"
    },
    {
        "name": "Ds005602Train_Ds004469test_Left",
        "train": "Model/Ds005602Train_Ds004469test/left/ds005602_left_full_augmented_xyz_coords.csv",
        "test": "Model/Ds005602Train_Ds004469test/left/ds004469_left_coef_features_xyz_coords.csv",
        "out": "Model/Ds005602Train_Ds004469test/left/PointNet_Results"
    },
    {
        "name": "Ds005602Train_Ds004469test_Right",
        "train": "Model/Ds005602Train_Ds004469test/right/ds005602_right_full_augmented_xyz_coords.csv",
        "test": "Model/Ds005602Train_Ds004469test/right/ds004469_right_coef_features_xyz_coords.csv",
        "out": "Model/Ds005602Train_Ds004469test/right/PointNet_Results"
    },
    {
        "name": "All_Augment_Left",
        "train": "Model/All_Augment_tain/left/All_left_Train_augmented_xyz_coords.csv",
        "test": "Model/All_Augment_tain/left/All_left_coef_features_test_xyz_coords.csv",
        "out": "Model/All_Augment_tain/left/PointNet_Results"
    },
    {
        "name": "All_Augment_Right",
        "train": "Model/All_Augment_tain/right/All_right_Train_augmented_xyz_coords.csv",
        "test": "Model/All_Augment_tain/right/All_right_coef_features_test_xyz_coords.csv",
        "out": "Model/All_Augment_tain/right/PointNet_Results"
    }
]

for t in tasks:
    print(f"===========================================================")
    print(f"Running PointNet Training for: {t['name']}")
    print(f"===========================================================")
    
    cmd = [
        "python", "Model/train_pointnet_generic.py",
        "--train_csv", t["train"],
        "--test_csv", t["test"],
        "--output_dir", t["out"],
        "--num_points", "840"
    ]
    
    subprocess.run(cmd)
    
print("All tasks completed.")
