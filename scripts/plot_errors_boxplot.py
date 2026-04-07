#!/usr/bin/env python3
import glob
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def main():
    parser = argparse.ArgumentParser(description='Generates Boxplot of final landing errors.')
    # Flags existentes
    parser.add_argument('--hide_outliers', action='store_true', help='Physically removes outliers from the dataset (alters math)')
    parser.add_argument('--exclude_runs', nargs='*', default=[], help='Runs to exclude in ANGLE:RUN format (e.g., 0.0:4 30.0:18)')
    # Novas flags visuais
    parser.add_argument('--hide_visual_outliers', action='store_true', help='Hides outliers visually but keeps them in the math for quartiles/mean')
    parser.add_argument('--ymax', type=float, default=None, help='Manually sets the maximum Y-axis limit to crop the image')
    args = parser.parse_args()

    # Processa a lista de exclusões manuais
    exclusions = []
    for ex in args.exclude_runs:
        try:
            ang_str, run_str = ex.split(':')
            exclusions.append((round(float(ang_str), 1), int(run_str)))
        except ValueError:
            print(f"Warning: Invalid exclusion format for '{ex}'. Please use ANGLE:RUN (e.g., 30.0:5).")

    csv_files = glob.glob("log_ang_*_run_*.csv")
    if not csv_files:
        print("No CSV files found.")
        return

    error_data = {}

    for file in csv_files:
        match = re.search(r'log_ang_([0-9.]+)_run_([0-9]+)\.csv', file)
        if match:
            angle_rad = float(match.group(1))
            angle_deg = round(np.degrees(angle_rad), 1)
            run_num = int(match.group(2))
            
            if (angle_deg, run_num) in exclusions:
                print(f"[MANUALLY EXCLUDED] Angle: {angle_deg}° | Run: {run_num}")
                continue

            try:
                df = pd.read_csv(file)
                if 'final_error_projected' in df.columns:
                    final_error_cm = df['final_error_projected'].iloc[-1] * 100
                    if angle_deg not in error_data:
                        error_data[angle_deg] = []
                    error_data[angle_deg].append({'run': run_num, 'val': final_error_cm})
            except Exception as e:
                print(f"Error reading {file}: {e}")

    sorted_angles = sorted(error_data.keys())
    data_to_plot = []

    print("================ ERROR PROCESSING ================")
    for angle in sorted_angles:
        items = error_data[angle]
        vals = np.array([item['val'] for item in items])
        
        # Exclusão estatística severa (--hide_outliers)
        if args.hide_outliers and len(vals) > 0:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            valid_vals = []
            for item in items:
                if lower_bound <= item['val'] <= upper_bound:
                    valid_vals.append(item['val'])
                else:
                    print(f"[PHYSICALLY REMOVED] Angle: {angle:^4}° | Run: {item['run']:^2} | Error: {item['val']:.2f} cm")
            data_to_plot.append(valid_vals)
        else:
            data_to_plot.append(vals)

    plt.figure(figsize=(10, 6))
    
    # Verifica se os fliers (pontinhos) devem ser exibidos
    show_fliers = not (args.hide_outliers or args.hide_visual_outliers)

    box = plt.boxplot(data_to_plot, patch_artist=True, 
                      showfliers=show_fliers,
                      showmeans=True, meanline=True, 
                      meanprops={'color': 'red', 'linewidth': 1.5})
    
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Legenda em Inglês
    legend_handles = [
        mpatches.Patch(color='lightblue', label='Box (IQR: 25% to 75%)', alpha=0.7),
        mlines.Line2D([], [], color='orange', linewidth=1.5, label='Median'),
        mlines.Line2D([], [], color='red', linestyle='--', linewidth=1.5, label='Mean'),
        mlines.Line2D([], [], color='black', linewidth=1.5, label='Whiskers (1.5x IQR)')
    ]
    if show_fliers:
        legend_handles.append(mlines.Line2D([], [], color='white', marker='o', 
                              markerfacecolor='white', markeredgecolor='black', markersize=6, label='Outliers'))

    plt.legend(handles=legend_handles, loc='upper left', fontsize=10)

    plt.xticks(range(1, len(sorted_angles) + 1), [f"{a}°" for a in sorted_angles])
    plt.title('Landing Final Error Dispersion by Platform Inclination', fontsize=14)
    plt.xlabel('Platform Inclination [deg]', fontsize=12)
    plt.ylabel('Projected Final Error [cm]', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Aplica o corte do eixo Y, se a flag foi utilizada
    if args.ymax is not None:
        plt.ylim(top=args.ymax, bottom=0.0)

    output_file = "boxplot_erros_finais.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    
    if args.hide_visual_outliers:
        print("[VISUAL] Outliers hidden visually, but dataset remains intact.")
    if args.ymax:
        print(f"[VISUAL] Y-Axis limited to maximum: {args.ymax}")
        
    print(f"========================================================\nPlot saved as: {output_file}\n")

if __name__ == '__main__':
    main()
