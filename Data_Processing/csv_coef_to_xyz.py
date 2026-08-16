import pandas as pd
import numpy as np
import scipy.special as sp
import argparse
import os

def evaluate_spharm(coeffs_list, L, theta_grid, phi_grid):
    THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    X = np.zeros_like(THETA)
    Y = np.zeros_like(THETA)
    Z = np.zeros_like(THETA)
    idx = 0
    for l in range(L + 1):
        c = coeffs_list[idx]
        y = sp.sph_harm_y(l, 0, THETA, PHI).real
        X += c[0] * y; Y += c[1] * y; Z += c[2] * y
        idx += 1
        for m in range(1, l + 1):
            cr = coeffs_list[idx]; idx += 1
            ci = coeffs_list[idx]; idx += 1
            y_comp = sp.sph_harm_y(l, m, THETA, PHI)
            factor = np.sqrt(2)
            X += factor * (cr[0] * y_comp.real + ci[0] * y_comp.imag)
            Y += factor * (cr[1] * y_comp.real + ci[1] * y_comp.imag)
            Z += factor * (cr[2] * y_comp.real + ci[2] * y_comp.imag)
    return X, Y, Z

def process_csv(input_csv, theta_step=9.0, phi_step=9.0):
    print(f"Processing {input_csv} ...")
    df = pd.read_csv(input_csv)
    
    coef_cols = [c for c in df.columns if c.startswith('Coef_')]
    if not coef_cols:
        print("No Coef_ columns found. Skipping.")
        return
        
    num_coeffs_flat = len(coef_cols)
    num_coeffs = num_coeffs_flat // 3
    L = int(np.sqrt(num_coeffs)) - 1
    
    theta_deg = np.linspace(0, 180, int(180/theta_step) + 1)
    phi_deg = np.linspace(0, 360, int(360/phi_step) + 1)[:-1]
    num_points = len(theta_deg) * len(phi_deg)
    
    print(f"L={L}, Coeffs={num_coeffs}. Grid size: {num_points} points.")
    
    output_rows = []
    
    for idx, row in df.iterrows():
        flat_coeffs = row[coef_cols].values.astype(float)
        coeffs = flat_coeffs.reshape(-1, 3)
        
        X, Y, Z = evaluate_spharm(coeffs, L, np.radians(theta_deg), np.radians(phi_deg))
        
        flat_pts = np.empty(num_points * 3)
        flat_pts[0::3] = X.flatten()
        flat_pts[1::3] = Y.flatten()
        flat_pts[2::3] = Z.flatten()
        
        meta_dict = row.drop(coef_cols).to_dict()
        
        rounded_pts = [float("{:.8f}".format(val)) for val in flat_pts]
        out_row = list(meta_dict.values()) + rounded_pts
        output_rows.append(out_row)
        
    meta_cols = list(df.drop(columns=coef_cols).columns)
    header = meta_cols.copy()
    for i in range(num_points):
        header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
        
    out_df = pd.DataFrame(output_rows, columns=header)
    
    output_csv = input_csv.replace('.csv', '')
    if not output_csv.endswith('_xyz_coords'):
        output_csv += '_xyz_coords'
    output_csv += '.csv'
    
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', help="CSV files to process", required=True)
    args = parser.parse_args()
    
    for f in args.files:
        if os.path.exists(f):
            process_csv(f)
        else:
            print(f"File not found: {f}")
