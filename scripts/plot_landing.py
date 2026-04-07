#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ==========================================
    # 1. Argument Parsing (Flags)
    # ==========================================
    parser = argparse.ArgumentParser(description='Generates landing plots for Beetle Omni from CSV.')
    
    # Input file
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file (e.g., log_ang_0.26_run_1.csv)')
    
    # Platform Data (from .world file)
    parser.add_argument('--plat_x', type=float, default=1.2, help='Platform X position (default: 1.2)')
    parser.add_argument('--plat_y', type=float, default=1.2, help='Platform Y position (default: 1.2)')
    parser.add_argument('--plat_z', type=float, default=0.6, help='Platform Z position (default: 0.6)')
    parser.add_argument('--pitch', type=float, default=0.0, help='Platform pitch inclination in radians (default: 0.0)')
    
    # Drone Data (from .xacro file)
    parser.add_argument('--cg_offset', type=float, default=0.1403, help='Z distance from CoG to leg contact point (default: 0.1403)')
    
    # Output filename
    parser.add_argument('--out', type=str, default='landing_analysis.png', help='Output image filename')

    args = parser.parse_args()

    # ==========================================
    # 2. Data Processing
    # ==========================================
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: File '{args.csv}' not found.")
        return

    # Filter only Phase 2 (Perpendicular Descent) and normalize time
    df_fase2 = df[df['phase'] == 2].copy()
    if df_fase2.empty:
        print("Error: No 'phase == 2' found in CSV.")
        return
        
    df_fase2['t_norm'] = df_fase2['time'] - df_fase2['time'].iloc[0]

    # --- TOUCHDOWN DETECTION ---
    # Find the moment Z position reaches its lower limit (physical landing)
    min_z = df_fase2['real_z'].min()
    try:
        idx_touchdown = df_fase2[df_fase2['real_z'] <= min_z + 0.005].index[0]
        t_touchdown = df_fase2.loc[idx_touchdown, 't_norm']
    except IndexError:
        print("Warning: Touchdown not detected by Z limits.")
        idx_touchdown = df_fase2.index[-1]
        t_touchdown = df_fase2['t_norm'].iloc[-1]

    # --- POST-TOUCHDOWN REFERENCE UPDATE ---
    # Initialize new reference columns by copying original ones
    df_fase2['new_ref_x'] = df_fase2['ref_x']
    df_fase2['new_ref_y'] = df_fase2['ref_y']
    df_fase2['new_ref_z'] = df_fase2['ref_z']

    # Calculate exact CoG position when drone rests on platform center
    # Projecting CoG offset based on ramp pitch
    target_cg_x = args.plat_x - (args.cg_offset * np.sin(args.pitch))
    target_cg_y = args.plat_y
    target_cg_z = args.plat_z + (args.cg_offset * np.cos(args.pitch))

    # Apply new target starting from touchdown (pressure phase in simulation)
    pos_toque = df_fase2.index >= idx_touchdown
    df_fase2.loc[pos_toque, 'new_ref_x'] = target_cg_x
    df_fase2.loc[pos_toque, 'new_ref_y'] = target_cg_y
    df_fase2.loc[pos_toque, 'new_ref_z'] = target_cg_z

    # --- ACTUAL ABSOLUTE ERROR CALCULATION ---
    df_fase2['e_x'] = np.abs(df_fase2['real_x'] - df_fase2['new_ref_x'])
    df_fase2['e_y'] = np.abs(df_fase2['real_y'] - df_fase2['new_ref_y'])
    df_fase2['e_z'] = np.abs(df_fase2['real_z'] - df_fase2['new_ref_z'])
    df_fase2['e_total'] = np.sqrt(df_fase2['e_x']**2 + df_fase2['e_y']**2 + df_fase2['e_z']**2)

    # ==========================================
    # 3. Plot Generation (Academic Style)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    pitch_deg = np.degrees(args.pitch)

    # Convert time column to numpy array
    t = df_fase2['t_norm'].to_numpy()

    # Plot 1: Z-Axis Tracking
    axes[0].plot(t, df_fase2['new_ref_z'].to_numpy(), label='Post-Touchdown Ref (Z)', linestyle='--', color='blue', linewidth=2)
    axes[0].plot(t, df_fase2['real_z'].to_numpy(), label='Actual (Z Drone)', color='red', linewidth=2)
    axes[0].axvline(x=t_touchdown, color='black', linestyle='-.', label='Touchdown')
    axes[0].set_title(f'Z-Axis Descent Trajectory (Ramp Pitch: {pitch_deg:.1f}°)', fontsize=14)
    axes[0].set_ylabel('Z Position [m]', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot 2: X and Y Axis Tracking
    axes[1].plot(t, df_fase2['new_ref_x'].to_numpy(), label='Corrected Ref X', linestyle='--', color='blue')
    axes[1].plot(t, df_fase2['real_x'].to_numpy(), label='Actual X', color='cyan', linewidth=2)
    axes[1].plot(t, df_fase2['new_ref_y'].to_numpy(), label='Corrected Ref Y', linestyle='--', color='red')
    axes[1].plot(t, df_fase2['real_y'].to_numpy(), label='Actual Y', color='orange', linewidth=2)
    axes[1].axvline(x=t_touchdown, color='black', linestyle='-.')
    axes[1].set_title('Horizontal Tracking (X and Y Planes)', fontsize=14)
    axes[1].set_ylabel('Position [m]', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Absolute Error Evolution
    axes[2].plot(t, df_fase2['e_x'].to_numpy(), label='Error X', color='blue', alpha=0.5)
    axes[2].plot(t, df_fase2['e_y'].to_numpy(), label='Error Y', color='orange', alpha=0.5)
    axes[2].plot(t, df_fase2['e_z'].to_numpy(), label='Error Z', color='purple', alpha=0.5)
    axes[2].plot(t, df_fase2['e_total'].to_numpy(), label='Total Error (3D Norm)', color='brown', linewidth=2.5)
    axes[2].axvline(x=t_touchdown, color='black', linestyle='-.')
    
    # Get final error (last row)
    err_final = df_fase2['e_total'].iloc[-1]
    axes[2].set_title(f'Absolute Error Evolution (Final Positioning Error: {err_final*100:.2f} cm)', fontsize=14)
    axes[2].set_xlabel('Phase 2 Time [s]', fontsize=12)
    axes[2].set_ylabel('Error [m]', fontsize=12)
    axes[2].legend(loc='best')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"\n[SUCCESS] Plot generated and saved as: {args.out}")
    print(f"Final Positioning Error (Stabilized): {err_final*100:.2f} cm\n")

if __name__ == '__main__':
    main()
