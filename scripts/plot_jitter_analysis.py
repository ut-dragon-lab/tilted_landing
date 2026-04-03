#!/usr/bin/env python
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
    parser = argparse.ArgumentParser(description='Analisa o Jitter de controle (Desvio Padrão por Run).')
    parser.add_argument('--hide_outliers', action='store_true', help='Remove fisicamente os outliers do Boxplot Global')
    args = parser.parse_args()

    bag_files = glob.glob("jitter_ang_*_run_*.bag")
    if not bag_files:
        print("Nenhum arquivo Bag encontrado.")
        return

    output_dir = "graficos_jitter_individuais"
    os.makedirs(output_dir, exist_ok=True)

    # Dicionário guardará o DP de cada run: {angle: [{'run': run_num, 'std': dp}, ...]}
    jitter_std_data = {}

    print("================ PROCESSAMENTO DE JITTER (DP) ===============")
    for file in bag_files:
        match = re.search(r'jitter_ang_([0-9.]+)_run_([0-9]+)\.bag', file)
        if not match:
            continue
            
        angle_rad = float(match.group(1))
        angle_deg = round(np.degrees(angle_rad), 1)
        run_num = int(match.group(2))
        
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
        # Calcula dt em milissegundos
        dt_ms = np.diff(times) * 1000.0
        
        # === A MÁGICA ACONTECE AQUI ===
        # Calcula o Desvio Padrão (Jitter real) dessa única run
        run_std = np.std(dt_ms)
        
        if angle_deg not in jitter_std_data:
            jitter_std_data[angle_deg] = []
        jitter_std_data[angle_deg].append({'run': run_num, 'std': run_std})

        # --- Gráfico Individual (mantemos mostrando os pontos no tempo) ---
        time_elapsed = times[1:] - times[0]
        plt.figure(figsize=(10, 4))
        plt.plot(time_elapsed, dt_ms, color='teal', linewidth=1.2)
        mean_dt = np.mean(dt_ms)
        plt.axhline(mean_dt, color='red', linestyle='--', label=f'Média: {mean_dt:.2f} ms')
        plt.title(f'Sinal de Controle (Inclinação: {angle_deg}°, Run: {run_num} | DP: {run_std:.2f} ms)', fontsize=12)
        plt.xlabel('Tempo de Voo (s)', fontsize=10)
        plt.ylabel('Intervalo entre Comandos (ms)', fontsize=10)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"jitter_line_ang_{angle_deg}_run_{run_num}.png"), dpi=200)
        plt.close()

    # --- Gera o Boxplot Global do Desvio Padrão ---
    sorted_angles = sorted(jitter_std_data.keys())
    data_to_plot = []

    for angle in sorted_angles:
        items = jitter_std_data[angle]
        # Extrai os DPs das runs desse ângulo
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
                    print(f"[OUTLIER EXCLUÍDO] Ângulo: {angle:^4}° | Run: {item['run']:^2} | DP: {item['std']:.2f} ms")
            data_to_plot.append(valid_stds)
        else:
            data_to_plot.append(stds)

    plt.figure(figsize=(10, 6))
    
    box = plt.boxplot(data_to_plot, patch_artist=True, 
                      showfliers=not args.hide_outliers,
                      showmeans=True, meanline=True, 
                      meanprops={'color': 'red', 'linewidth': 1.5})
    
    for patch in box['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)

    # Legenda Customizada
    legend_handles = [
        mpatches.Patch(color='lightcoral', label='Caixa (IQR: 25% a 75%)', alpha=0.7),
        mlines.Line2D([], [], color='orange', linewidth=1.5, label='Mediana'),
        mlines.Line2D([], [], color='red', linestyle='--', linewidth=1.5, label='Média'),
        mlines.Line2D([], [], color='black', linewidth=1.5, label='Bigodes (1.5x IQR)')
    ]
    if not args.hide_outliers:
        legend_handles.append(mlines.Line2D([], [], color='white', marker='o', 
                              markerfacecolor='white', markeredgecolor='black', markersize=6, label='Outliers'))

    plt.legend(handles=legend_handles, loc='upper left', fontsize=10)

    plt.xticks(range(1, len(sorted_angles) + 1), [f"{a}°" for a in sorted_angles])
    plt.title('Variabilidade do NMPC: Desvio Padrão do Tempo de Execução por Run', fontsize=14)
    plt.xlabel('Inclinação da Plataforma', fontsize=12)
    plt.ylabel('Desvio Padrão do Ciclo de Controle (ms)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    output_file = "boxplot_jitter_dp_global.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"===============================================================\nGráfico de Boxplot do Jitter (DP) salvo como: {output_file}\n")

if __name__ == '__main__':
    main()
