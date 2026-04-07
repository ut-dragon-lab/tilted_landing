#!/bin/bash

# Força o uso do ponto como separador decimal para evitar problemas com o awk
export LC_NUMERIC=C

if [ "$#" -eq 0 ]; then
    echo "Uso: $0 <angulo_rad>:<run_idx> [<angulo_rad>:<run_idx> ...]"
    echo "Exemplo: $0 0.0:4 0.5236:18"
    exit 1
fi

PLAT_X="1.0"
PLAT_Y="1.0"
PLAT_Z="0.6"
CG_OFFSET="0.1403"

# Arrays com os ruidos gaussianos pre-calculados (N = 20, mu = 0, sigma = 0.15)
NOISE_X=( 0.12 -0.05  0.08 -0.15  0.03  0.22 -0.11  0.01 -0.09  0.17 -0.21  0.06 -0.02  0.14 -0.18  0.09 -0.04  0.11 -0.13  0.05)
NOISE_Y=(-0.08  0.14 -0.02 -0.11  0.09  0.04 -0.18  0.21 -0.06 -0.12  0.15  0.01 -0.09  0.19  0.07 -0.14  0.11 -0.03 -0.20  0.05)

WORLD_FILE="$HOME/ros/jsk_aerial_robot_ws/src/jsk_aerial_robot_dev/aerial_robot_simulation/gazebo_model/world/0066.world"
TOPIC_PREFIX="/beetle1"

# O loop iterará sobre cada par "angulo:run" passado na linha de comando
for PAIR in "$@"; do
    # Extrai o ângulo (pitch) e o número da run separando pelo caractere ":"
    PITCH=$(echo $PAIR | cut -d':' -f1)
    RUN=$(echo $PAIR | cut -d':' -f2)

    echo "================================================="
    echo "Executando Run Especifica -> Angulo: $PITCH rad | Run: $RUN"
    echo "================================================="

    # Verifica se a run está no limite dos vetores de ruído (0 a 19)
    if [ "$RUN" -lt 0 ] || [ "$RUN" -ge 20 ]; then
        echo "Erro: A run $RUN esta fora do limite (0-19). Pulando..."
        continue
    fi

    # Calcula a nova posicao alvo somando o ruido
    TARGET_X=$(awk "BEGIN {print 1.0 + ${NOISE_X[$RUN]}}")
    TARGET_Y=$(awk "BEGIN {print 1.0 + ${NOISE_Y[$RUN]}}")
    TARGET_Z="1.5" # Z fixo para manter a altitude de hover

    sed -i -E "s/<static>true<\/static> <pose>.*<\/pose>/<static>true<\/static> <pose>$PLAT_X $PLAT_Y $PLAT_Z 0 $PITCH 0<\/pose>/g" $WORLD_FILE

    roslaunch beetle_omni bringup_nmpc_omni.launch real_machine:=false simulation:=True headless:=False nmpc_mode:=0 end_effector:=downward_cam world_type:=1 &
    SIM_PID=$!

    echo "Aguardando 15s para o Gazebo inicializar..."
    sleep 15
    
    roslaunch tilted_landing apriltag_sim.launch & 
    SIM_PID2=$!

    echo "Aguardando 5s para o nó da apriltag inicializar..."
    sleep 5

    echo "Enviando comando START (ARM)..."
    for i in {1..3}; do
        rostopic pub -1 ${TOPIC_PREFIX}/teleop_command/start std_msgs/Empty "{}"
        sleep 0.5
    done
    sleep 2

    echo "Enviando comando TAKEOFF..."
    for i in {1..3}; do
        rostopic pub -1 ${TOPIC_PREFIX}/teleop_command/takeoff std_msgs/Empty "{}"
        sleep 0.5
    done
    sleep 8 

    # Inicia o gravador ROSBAG
    BAG_NAME="jitter_ang_${PITCH}_run_${RUN}.bag"
    rosbag record -O $BAG_NAME ${TOPIC_PREFIX}/four_axes/command __name:=jitter_bag_recorder &
    
    echo "Navegando para o ponto alvo (X: $TARGET_X, Y: $TARGET_Y, Z: $TARGET_Z)..."
    rosrun tilted_landing omni_nav.py --mode mission --target "$TARGET_X $TARGET_Y $TARGET_Z"
    sleep 5 

    echo "Iniciando pouso..."
    CSV_NAME="log_ang_${PITCH}_run_${RUN}.csv"
    rosrun tilted_landing omni_land.py "$CSV_NAME" $PITCH
    sleep 5

    # Para o rosbag graciosamente
    rosnode kill /jitter_bag_recorder
    sleep 2

    # =========================================================
    # GERAÇÃO DOS GRÁFICOS (mantido do seu auto_exp original)
    # =========================================================
    echo "Gerando graficos da rodada..."
    PNG_NAME="analise_ang_${PITCH}_run_${RUN}.png"
    rosrun tilted_landing plot_landing.py --csv "$CSV_NAME" \
                            --plat_x $PLAT_X \
                            --plat_y $PLAT_Y \
                            --plat_z $PLAT_Z \
                            --pitch -$PITCH \
                            --cg_offset $CG_OFFSET \
                            --out "$PNG_NAME"

    echo "Limpando ambiente..."
    kill $SIM_PID $SIM_PID2
    killall -9 $(pgrep rviz) gzserver gzclient roslaunch rosmaster
    sleep 5

done

echo "Todas as runs especificas foram concluidas."
