#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ==========================================
    # 1. Configuração dos Argumentos (Flags)
    # ==========================================
    parser = argparse.ArgumentParser(description='Gera gráficos de pouso do Beetle Omni extraídos do CSV.')
    
    # Arquivo de entrada
    parser.add_argument('--csv', type=str, required=True, help='Caminho para o arquivo CSV (ex: log_ang_0.26_run_1.csv)')
    
    # Dados da Plataforma (do arquivo .world)
    parser.add_argument('--plat_x', type=float, default=1.2, help='Posição X da plataforma (default: 1.2)')
    parser.add_argument('--plat_y', type=float, default=1.2, help='Posição Y da plataforma (default: 1.2)')
    parser.add_argument('--plat_z', type=float, default=0.6, help='Posição Z da plataforma (default: 0.6)')
    parser.add_argument('--pitch', type=float, default=0.0, help='Inclinação (Pitch) da plataforma em radianos (default: 0.0)')
    
    # Dados do Drone (do arquivo .xacro)
    parser.add_argument('--cg_offset', type=float, default=0.1403, help='Distância Z do CoG até o ponto de contato da perna (default: 0.1403)')
    
    # Nome do arquivo de saída
    parser.add_argument('--out', type=str, default='analise_pouso.png', help='Nome da imagem a ser salva')

    args = parser.parse_args()

    # ==========================================
    # 2. Processamento dos Dados
    # ==========================================
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{args.csv}' não encontrado.")
        return

    # Filtra apenas a Fase 2 (Descida Perpendicular) e normaliza o tempo
    df_fase2 = df[df['phase'] == 2].copy()
    if df_fase2.empty:
        print("Erro: Nenhuma 'phase == 2' encontrada no CSV.")
        return
        
    df_fase2['t_norm'] = df_fase2['time'] - df_fase2['time'].iloc[0]

    # --- DETECÇÃO DO TOUCHDOWN ---
    # Encontra o momento em que a posição Z atinge o limite inferior (pousou fisicamente)
    min_z = df_fase2['real_z'].min()
    try:
        idx_touchdown = df_fase2[df_fase2['real_z'] <= min_z + 0.005].index[0]
        t_touchdown = df_fase2.loc[idx_touchdown, 't_norm']
    except IndexError:
        print("Aviso: Touchdown não detectado pelos limites de Z.")
        idx_touchdown = df_fase2.index[-1]
        t_touchdown = df_fase2['t_norm'].iloc[-1]

    # --- ATUALIZAÇÃO DA REFERÊNCIA PÓS-TOUCHDOWN ---
    # Inicia as novas colunas de referência copiando as originais
    df_fase2['new_ref_x'] = df_fase2['ref_x']
    df_fase2['new_ref_y'] = df_fase2['ref_y']
    df_fase2['new_ref_z'] = df_fase2['ref_z']

    # Calcula a posição exata do CoG quando o drone repousa sobre o centro da plataforma
    # Projetando o offset do CoG baseado no pitch (inclinação) da rampa
    target_cg_x = args.plat_x - (args.cg_offset * np.sin(args.pitch))
    target_cg_y = args.plat_y
    target_cg_z = args.plat_z + (args.cg_offset * np.cos(args.pitch))

    # Aplica o novo alvo a partir do touchdown (fase de pressão no simulador)
    pos_toque = df_fase2.index >= idx_touchdown
    df_fase2.loc[pos_toque, 'new_ref_x'] = target_cg_x
    df_fase2.loc[pos_toque, 'new_ref_y'] = target_cg_y
    df_fase2.loc[pos_toque, 'new_ref_z'] = target_cg_z

    # --- CÁLCULO DO ERRO ABSOLUTO REAL ---
    df_fase2['e_x'] = np.abs(df_fase2['real_x'] - df_fase2['new_ref_x'])
    df_fase2['e_y'] = np.abs(df_fase2['real_y'] - df_fase2['new_ref_y'])
    df_fase2['e_z'] = np.abs(df_fase2['real_z'] - df_fase2['new_ref_z'])
    df_fase2['e_total'] = np.sqrt(df_fase2['e_x']**2 + df_fase2['e_y']**2 + df_fase2['e_z']**2)

    # ==========================================
    # 3. Geração dos Gráficos (Estética Científica)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    pitch_deg = np.degrees(args.pitch)

    # Converte a coluna de tempo para numpy array de uma vez
    t = df_fase2['t_norm'].to_numpy()

    # Gráfico 1: Acompanhamento Eixo Z
    axes[0].plot(t, df_fase2['new_ref_z'].to_numpy(), label='Referência Pós-Contato (Z)', linestyle='--', color='blue', linewidth=2)
    axes[0].plot(t, df_fase2['real_z'].to_numpy(), label='Real (Z Drone)', color='red', linewidth=2)
    axes[0].axvline(x=t_touchdown, color='black', linestyle='-.', label='Touchdown')
    axes[0].set_title(f'Trajetória de Descida no Eixo Z (Pitch Rampa: {pitch_deg:.1f}°)', fontsize=14)
    axes[0].set_ylabel('Posição Z (m)', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Gráfico 2: Acompanhamento Eixos X e Y
    axes[1].plot(t, df_fase2['new_ref_x'].to_numpy(), label='Ref Corrigida X', linestyle='--', color='blue')
    axes[1].plot(t, df_fase2['real_x'].to_numpy(), label='Real X', color='cyan', linewidth=2)
    axes[1].plot(t, df_fase2['new_ref_y'].to_numpy(), label='Ref Corrigida Y', linestyle='--', color='red')
    axes[1].plot(t, df_fase2['real_y'].to_numpy(), label='Real Y', color='orange', linewidth=2)
    axes[1].axvline(x=t_touchdown, color='black', linestyle='-.')
    axes[1].set_title('Acompanhamento Horizontal (Planos X e Y)', fontsize=14)
    axes[1].set_ylabel('Posição (m)', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Gráfico 3: Evolução do Erro Absoluto
    axes[2].plot(t, df_fase2['e_x'].to_numpy(), label='Erro X', color='blue', alpha=0.5)
    axes[2].plot(t, df_fase2['e_y'].to_numpy(), label='Erro Y', color='orange', alpha=0.5)
    axes[2].plot(t, df_fase2['e_z'].to_numpy(), label='Erro Z', color='purple', alpha=0.5)
    axes[2].plot(t, df_fase2['e_total'].to_numpy(), label='Erro Total (Norma 3D)', color='brown', linewidth=2.5)
    axes[2].axvline(x=t_touchdown, color='black', linestyle='-.')
    
    # Pega o erro final (última linha)
    err_final = df_fase2['e_total'].iloc[-1]
    axes[2].set_title(f'Evolução do Erro Absoluto (Erro de Posicionamento Final: {err_final*100:.2f} cm)', fontsize=14)
    axes[2].set_xlabel('Tempo na Fase 2 (s)', fontsize=12)
    axes[2].set_ylabel('Erro (m)', fontsize=12)
    axes[2].legend(loc='best')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"\n[SUCESSO] Gráfico gerado e salvo como: {args.out}")
    print(f"Erro Final de Posicionamento (Estabilizado): {err_final*100:.2f} cm\n")

if __name__ == '__main__':
    main()
