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
    parser = argparse.ArgumentParser(description='Gera Boxplot dos erros finais de pouso.')
    parser.add_argument('--hide_outliers', action='store_true', help='Remove fisicamente os outliers do conjunto de dados')
    args = parser.parse_args()

    csv_files = glob.glob("log_ang_*_run_*.csv")
    if not csv_files:
        print("Nenhum arquivo CSV encontrado.")
        return

    # Guarda dicionários no formato: {angle: [{'run': X, 'val': Y}, ...]}
    error_data = {}

    for file in csv_files:
        match = re.search(r'log_ang_([0-9.]+)_run_([0-9]+)\.csv', file)
        if match:
            angle_rad = float(match.group(1))
            angle_deg = round(np.degrees(angle_rad), 1)
            run_num = int(match.group(2))
            
            try:
                df = pd.read_csv(file)
                if 'final_error_projected' in df.columns:
                    final_error_cm = df['final_error_projected'].iloc[-1] * 100
                    if angle_deg not in error_data:
                        error_data[angle_deg] = []
                    error_data[angle_deg].append({'run': run_num, 'val': final_error_cm})
            except Exception as e:
                print(f"Erro ao ler {file}: {e}")

    sorted_angles = sorted(error_data.keys())
    data_to_plot = []

    print("================ PROCESSAMENTO DE ERROS ================")
    for angle in sorted_angles:
        items = error_data[angle]
        vals = np.array([item['val'] for item in items])
        
        if args.hide_outliers and len(vals) > 0:
            # Calcula limites baseados na cerca de Tukey
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            valid_vals = []
            for item in items:
                if lower_bound <= item['val'] <= upper_bound:
                    valid_vals.append(item['val'])
                else:
                    print(f"[OUTLIER EXCLUÍDO] Ângulo: {angle:^4}° | Run: {item['run']:^2} | Erro: {item['val']:.2f} cm")
            data_to_plot.append(valid_vals)
        else:
            data_to_plot.append(vals)

    plt.figure(figsize=(10, 6))
    
    # O Matplotlib agora fará o plot *apenas* com os dados validados
    box = plt.boxplot(data_to_plot, patch_artist=True, 
                      showfliers=not args.hide_outliers,
                      showmeans=True, meanline=True, 
                      meanprops={'color': 'red', 'linewidth': 1.5})
    
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # --- Criação da Legenda Customizada ---
    legend_handles = [
        mpatches.Patch(color='lightblue', label='Caixa (IQR: 25% a 75%)', alpha=0.7),
        mlines.Line2D([], [], color='orange', linewidth=1.5, label='Mediana'),
        mlines.Line2D([], [], color='red', linestyle='--', linewidth=1.5, label='Média'),
        mlines.Line2D([], [], color='black', linewidth=1.5, label='Bigodes (1.5x IQR)')
    ]
    if not args.hide_outliers:
        legend_handles.append(mlines.Line2D([], [], color='white', marker='o', 
                              markerfacecolor='white', markeredgecolor='black', markersize=6, label='Outliers'))

    plt.legend(handles=legend_handles, loc='upper left', fontsize=10)

    plt.xticks(range(1, len(sorted_angles) + 1), [f"{a}°" for a in sorted_angles])
    plt.title('Dispersão do Erro Final de Pouso por Inclinação da Plataforma', fontsize=14)
    plt.xlabel('Inclinação da Plataforma', fontsize=12)
    plt.ylabel('Erro Final Projetado (cm)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    output_file = "boxplot_erros_finais.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"========================================================\nGráfico salvo como: {output_file}\n")

if __name__ == '__main__':
    main()
