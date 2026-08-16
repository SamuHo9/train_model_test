import os
import glob
import csv
import re
import argparse
import tkinter as tk
from tkinter import filedialog
import numpy as np
import scipy.special as sp

# =============================================================================
# Helpers
# =============================================================================
def prompt_folder(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title, initialdir=os.getcwd())
    root.destroy()
    return folder if folder else None

def classify_subject(subject_name):
    is_left_side = subject_name.startswith("left_") or subject_name.startswith("lh_") or "_lh" in subject_name.lower()
    name_upper = subject_name.upper()
    
    if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
        return "Healthy Control", 0
    elif (is_left_side and "LEFT-TLE" in name_upper) or (not is_left_side and "RIGHT-TLE" in name_upper):
        return "Ipsilateral TLE (Diseased)", 1
    elif (is_left_side and "RIGHT-TLE" in name_upper) or (not is_left_side and "LEFT-TLE" in name_upper):
        return "Contralateral TLE (Healthy-side)", 2
    elif "TLE" in name_upper:
        return "Ipsilateral TLE (Diseased)", 1
    return "Unknown", -1

def parse_coef(filename):
    with open(filename, 'r') as f:
        content = f.read()
    pattern = re.compile(r"\{([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+)\}")
    matches = pattern.findall(content)
    coeffs = [[float(x) for x in m] for m in matches]
    num_match = re.search(r"\{\s*(\d+)", content)
    if num_match:
        num_coeffs = int(num_match.group(1))
        return coeffs[:num_coeffs]
    return coeffs

def evaluate_spharm(coeffs_list, L, theta_grid, phi_grid):
    THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    X = np.zeros_like(THETA)
    Y = np.zeros_like(THETA)
    Z = np.zeros_like(THETA)
    idx = 0
    for l in range(L + 1):
        c = coeffs_list[idx]
        y = sp.sph_harm(0, l, PHI, THETA).real
        X += c[0] * y; Y += c[1] * y; Z += c[2] * y
        idx += 1
        for m in range(1, l + 1):
            cr = coeffs_list[idx]; idx += 1
            ci = coeffs_list[idx]; idx += 1
            y_comp = sp.sph_harm(m, l, PHI, THETA)
            factor = np.sqrt(2)
            X += factor * (cr[0] * y_comp.real + ci[0] * y_comp.imag)
            Y += factor * (cr[1] * y_comp.real + ci[1] * y_comp.imag)
            Z += factor * (cr[2] * y_comp.real + ci[2] * y_comp.imag)
    return X, Y, Z

# =============================================================================
# Main Extraction
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spharm_dir", default=None, help="Path to spharm folder")
    parser.add_argument("--theta_step", type=float, default=9.0, help="Grid step size for polar angle")
    parser.add_argument("--phi_step", type=float, default=9.0, help="Grid step size for azimuthal angle")
    args = parser.parse_args()

    spharm_dir = args.spharm_dir
    if not spharm_dir:
        print("Opening folder picker...")
        spharm_dir = prompt_folder("Select folder containing .coef files")
    
    if not spharm_dir:
        return

    coef_files = sorted(glob.glob(os.path.join(spharm_dir, "*.coef")))
    if not coef_files:
        print(f"Error: No .coef files found in {spharm_dir}")
        return

    print(f"Found {len(coef_files)} coefficient files. Processing...")

    # Define grid
    theta_deg = np.linspace(0, 180, int(180/args.theta_step) + 1)
    phi_deg = np.linspace(0, 360, int(360/args.phi_step) + 1)[:-1]
    num_points = len(theta_deg) * len(phi_deg)
    print(f"Grid size: {len(theta_deg)}x{len(phi_deg)} = {num_points} points per mesh")

    ml_output_dir = os.path.join(os.path.dirname(spharm_dir), "ml_features")
    os.makedirs(ml_output_dir, exist_ok=True)
    
    folder_basename = os.path.basename(spharm_dir.rstrip("\\/"))
    out_csv = os.path.join(ml_output_dir, f"{folder_basename}_xyz_from_coef.csv")

    header = ["Subject", "Group_Name", "Group_Label", "BinaryClass", "DataType"]
    for idx in range(num_points):
        header.extend([f"x_{idx}", f"y_{idx}", f"z_{idx}"])

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filepath in coef_files:
            filename = os.path.basename(filepath)
            subject_name = filename.replace("_SPHARM_ellalign.coef", "").replace("_SPHARM.coef", "").replace(".coef", "")
            group_name, group_label = classify_subject(subject_name)
            binary_class = 1 if group_label == 1 else 0

            coeffs = parse_coef(filepath)
            L = int(np.sqrt(len(coeffs))) - 1
            
            X, Y, Z = evaluate_spharm(coeffs, L, np.radians(theta_deg), np.radians(phi_deg))
            
            # Flatten to 1D arrays: X, Y, Z
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = Z.flatten()
            
            # Interleave x, y, z
            flat_pts = np.empty(num_points * 3)
            flat_pts[0::3] = X_flat
            flat_pts[1::3] = Y_flat
            flat_pts[2::3] = Z_flat

            row = [subject_name, group_name, group_label, binary_class, "generated_from_coef"]
            row.extend(["{:.6f}".format(val) for val in flat_pts])
            writer.writerow(row)

    print(f"Successfully saved {num_points} XYZ points for {len(coef_files)} subjects to: {out_csv}")
    print(f"\n*** IMPORTANT for PointNet ***")
    print(f"Make sure to update `num_points = {num_points}` in your train_pointnet.py")

if __name__ == "__main__":
    main()
