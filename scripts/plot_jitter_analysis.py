#!/usr/bin/env python3
import glob
import re
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import rosbag

def main():
    parser = argparse.ArgumentParser(description='Analyzes control jitter (Std Dev per Run).')
    # Flags Estatísticas
    parser.add_argument('--hide_outliers', action='store_true', help='Physically removes outliers from the Global Boxplot (alters math)')
    parser.add_argument('--exclude_runs', nargs='*', default=[], help='Runs to exclude in ANGLE:RUN format (e.g., 0.0:4 30.0:18)')
    # Flags Visuais
    parser.add_argument('--hide_visual_outliers', action='store_true', help='Hides outliers visually but keeps them in the math for quartiles/mean')
    parser.add_argument('--ymax', type=float, default=None, help='Manually sets the maximum Y-axis limit to crop the image')
    args = parser.parse_args()

    exclusions = []
    for ex in args.exclude_runs:
        try:
            ang_str, run_str = ex.split(':')
            exclusions.append((round(float(ang_str), 1), int(run_str)))
        except ValueError:
            print(f"Warning: Invalid exclusion format for '{ex}'. Please use ANGLE:RUN.")

    bag_files = glob.glob("jitter_ang_*_run_*.bag")
    if not bag_files:
        print("No Bag files found.")
        return

    output_dir = "graficos_jitter_individuais"
    os.makedirs(output_dir, exist_ok=True)

    jitter_std_data = {}

    print("================ JITTER PROCESSING (STD DEV) ===============")
    for file in bag_files:
        match = re.search(r'jitter_ang_([0-9.]+)_run_([0-9]+)\.bag', file)
        if not match:
            continue
            
        angle_rad = float(match.group(1))
        angle_deg = round(np.degrees(angle_rad), 1)
        run_num = int(match.group(2))
        
        # Checa exclusão
        if (angle_deg, run_num) in exclusions:
            print(f"[MANUALLY EXCLUDED] Angle: {angle_deg}° | Run: {run_num}")
            continue

        times = []
        try:
            bag = rosbag.Bag(file)
            for topic, msg, t in bag.read_messages():
                times.append(t.to_sec())
            bag.close()
        except Exception as e:
            continue
            
        if len(times) < 2:
            continue

        times = np.array(times)
        dt_ms = np.diff(times) * 1000.0
        
        run_std = np.std(dt_ms)
        
        if angle_deg not in jitter_std_data:
            jitter_std_data[angle_deg] = []
        jitter_std_data[angle_deg].append({'run': run_num, 'std': run_std})

        # Gráfico Individual
        time_elapsed = times[1:] - times[0]
        plt.figure(figsize=(10, 4))
        plt.plot(time_elapsed, dt_ms, color='teal', linewidth=1.2)
        mean_dt = np.mean(dt_ms)
        plt.axhline(mean_dt, color='red', linestyle='--', label=f'Mean: {mean_dt:.2f} ms')
        plt.title(f'Control Command Topic Jitter (Inclination: {angle_deg}°, Run: {run_num} | Std Dev: {run_std:.2f} ms)', fontsize=12)
        plt.xlabel('Flight Time [s]', fontsize=10)
        plt.ylabel('Command Interval [ms]', fontsize=10)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"jitter_line_ang_{angle_deg}_run_{run_num}.png"), dpi=200)
        plt.close()

    sorted_angles = sorted(jitter_std_data.keys())
    data_to_plot = []

    for angle in sorted_angles:
        items = jitter_std_data[angle]
        stds = np.array([item['std'] for item in items])
        
        if args.hide_outliers and len(stds) > 0:
            q1, q3 = np.percentile(stds, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            valid_stds = []
            for item in items:
                if lower_bound <= item['std'] <= upper_bound:
                    valid_stds.append(item['std'])
                else:
                    print(f"[PHYSICALLY REMOVED] Angle: {angle:^4}° | Run: {item['run']:^2} | Std Dev: {item['std']:.2f} ms")
            data_to_plot.append(valid_stds)
        else:
            data_to_plot.append(stds)

    plt.figure(figsize=(10, 6))
    
    # Verifica se os fliers (pontinhos) devem ser exibidos
    show_fliers = not (args.hide_outliers or args.hide_visual_outliers)

    box = plt.boxplot(data_to_plot, patch_artist=True, 
                      showfliers=show_fliers,
                      showmeans=True, meanline=True, 
                      meanprops={'color': 'red', 'linewidth': 1.5})
    
    for patch in box['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)

    # Legenda em Inglês
    legend_handles = [
        mpatches.Patch(color='lightcoral', label='Box (IQR: 25% to 75%)', alpha=0.7),
        mlines.Line2D([], [], color='orange', linewidth=1.5, label='Median'),
        mlines.Line2D([], [], color='red', linestyle='--', linewidth=1.5, label='Mean'),
        mlines.Line2D([], [], color='black', linewidth=1.5, label='Whiskers (1.5x IQR)')
    ]
    if show_fliers:
        legend_handles.append(mlines.Line2D([], [], color='white', marker='o', 
                              markerfacecolor='white', markeredgecolor='black', markersize=6, label='Outliers'))

    plt.legend(handles=legend_handles, loc='upper left', fontsize=10)

    plt.xticks(range(1, len(sorted_angles) + 1), [f"{a}°" for a in sorted_angles])
    
    # Títulos e Eixos mais honestos!
    plt.title('Control Command Topic Interval Jitter Variability by Platform Inclination', fontsize=14)
    plt.xlabel('Platform Inclination [deg]', fontsize=12)
    plt.ylabel('Command Interval Jitter Std. Dev. [ms]', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Aplica o corte do eixo Y, se a flag foi utilizada
    if args.ymax is not None:
        plt.ylim(bottom=0.0, top=args.ymax)

    output_file = "boxplot_jitter_dp_global.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    if args.hide_visual_outliers:
        print("[VISUAL] Outliers hidden visually, but dataset remains intact.")
    if args.ymax:
        print(f"[VISUAL] Y-Axis limited to maximum: {args.ymax}")

    print(f"===============================================================\nPlot saved as: {output_file}\n")

if __name__ == '__main__':
    main()
