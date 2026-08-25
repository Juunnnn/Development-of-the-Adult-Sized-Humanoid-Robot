# ══════════════════════════════════════════════════════════════
# [7] Teleop 함수
# ══════════════════════════════════════════════════════════════
ADB=/usr/bin/adb
TELEOP_ROOT=~/Development\ of\ the\ Adult\ Sized\ Humanoid\ Robot
CONTROL_DIR=$TELEOP_ROOT/Teleoperation/control
UTIL_DIR=$TELEOP_ROOT/Teleoperation/teleop_utility

xrteleop() {
    echo "[xrteleop] Conda env 'tv' 활성화..."
    conda activate tv
    export ROS_MASTER_URI=$ROS_MASTER_URI
    export ROS_IP=$ROS_IP
    cd "$CONTROL_DIR"
    $ADB reverse --remove tcp:8012 2>/dev/null
    $ADB reverse tcp:8012 tcp:8012
    echo "[xrteleop] ADB 포트포워딩 완료"
    python HR_teleop.py
    $ADB reverse --remove tcp:8012
    echo "[xrteleop] 종료"
}

xrgazebo() {
    if [[ "$ROS_MASTER_URI" != *"${STATION_LAN_IP}"* ]]; then
        echo "⚠️  실험 모드에서만 사용 가능. 'rme' 먼저 실행하세요."
        return 1
    fi
    conda deactivate 2>/dev/null
    source /opt/ros/noetic/setup.bash
    source ~/catkin_ws/devel/setup.bash
    export ROS_MASTER_URI=http://${STATION_LAN_IP}:11311
    export ROS_IP=${STATION_LAN_IP}
    roslaunch Wholebody_39_DoF_URDF gazebo.launch
}

xrrviz_gazebo() {
    if [[ "$ROS_MASTER_URI" != *"${GENE_WIFI_IP}"* ]]; then
        echo "⚠️  실제 로봇 모드에서만 사용 가능. 'rmr' 먼저 실행하세요."
        return 1
    fi
    conda deactivate 2>/dev/null
    roslaunch Wholebody_39_DoF_URDF display.launch
}

xrrviz_real() {
    if [[ "$ROS_MASTER_URI" != *"${GENE_WIFI_IP}"* ]]; then
        echo "⚠️  실제 로봇 모드에서만 사용 가능. 'rmr' 먼저 실행하세요."
        return 1
    fi
    conda deactivate 2>/dev/null
    roslaunch Wholebody_39_DoF_URDF display_real.launch
}

xrcamerastart() {
    fuser -k 8080/tcp 2>/dev/null
    sleep 1
    echo "[Camera] Jetson 카메라 프로세스 정리... (${CURRENT_JETSON_SSH})"
    ssh ${CURRENT_JETSON_SSH} "
        sudo pkill -9 -f realsense 2>/dev/null
        sudo pkill -9 -f nodelet 2>/dev/null
        sleep 2
        echo '[Jetson] cleanup done'
    "
    if ! rostopic list > /dev/null 2>&1; then
        echo "[Camera] roscore 시작..."
        roscore > /tmp/roscore.log 2>&1 &
        sleep 3
    fi
    echo "[Camera] Jetson RealSense 시작..."
    ssh ${CURRENT_JETSON_SSH} "
        source /opt/ros/noetic/setup.bash
        source ~/catkin_ws/devel/setup.bash
        export ROS_MASTER_URI=${ROS_MASTER_URI}
        export ROS_IP=${CURRENT_JETSON_IP}
        nohup roslaunch realsense2_camera rs_camera.launch \
            color_width:=640 color_height:=480 color_fps:=60 \
            > /tmp/camera.log 2>&1 &
    "
    echo "[Camera] 초기화 대기 (5s)..."
    sleep 5
    rosrun web_video_server web_video_server > /tmp/web_video.log 2>&1 &
    echo "[Camera] 스트림: http://localhost:8080/stream?topic=/camera/color/image_raw&type=mjpeg"
}

xrhandmujoco() {
    echo "[AmazingHand] MuJoCo Bridge 시작... (${CURRENT_JETSON_SSH})"
    ssh -t ${CURRENT_JETSON_SSH} "
        export MUJOCO_GL=egl
        export DISPLAY=:1
        export ROS_MASTER_URI=${ROS_MASTER_URI}
        export ROS_IP=${CURRENT_JETSON_IP}
        source /opt/ros/noetic/setup.bash
        source ~/catkin_ws/devel/setup.bash
        python3 ~/teleop_utility/hand_controller/ah_sim_right.py &
        python3 ~/teleop_utility/hand_controller/ah_sim_left.py &
        wait
    "
}

xrhandarduinogene() {
    echo "[AmazingHand] Arduino Control 시작..."
    ssh -t gene@${GENE_WIFI_IP} "
        export ROS_MASTER_URI=${ROS_MASTER_URI}
        export ROS_IP=${GENE_WIFI_IP}
        source /opt/ros/noetic/setup.bash
        sudo chmod 666 /dev/ttyACM0
        cd ~/teleop_utility/hand_controller
        python3 amazing_hand_arduino.py
    "
}

xrhandarduinojetson() {
    echo "[AmazingHand] Jetson Arduino Control 시작... (${CURRENT_JETSON_SSH})"
    ssh -t ${CURRENT_JETSON_SSH} "
        export ROS_MASTER_URI=${ROS_MASTER_URI}
        export ROS_IP=${CURRENT_JETSON_IP}
        source /opt/ros/noetic/setup.bash
        sudo chmod 666 /dev/ttyACM0
        cd ~/teleop_utility/hand_controller
        python3 amazing_hand_arduino.py
    "
}

xrneck() {
    echo "[Neck] Dynamixel Node 시작... (gene)"
    ssh -t gene@${GENE_WIFI_IP} "
        export ROS_MASTER_URI=${ROS_MASTER_URI}
        export ROS_IP=${GENE_WIFI_IP}
        source /opt/ros/noetic/setup.bash
        sudo chmod 666 /dev/ttyACM0
        cd ~/teleop_utility/neck_controller
        python3 neck_dynamixel_node.py
    "
}

# ── 그리퍼 on/off 원격 토글 ─────────────────────────────────
TELEOP_CONFIG_PATH="$HOME/Development of the Adult Sized Humanoid Robot/Teleoperation/control/config.py"
GENE_CONFIG_PATH="~/teleop_utility/neck_controller/config.py"

_set_gripper_local() {
    sed -i "s/^USE_GRIPPER = .*/USE_GRIPPER = $1/" "${TELEOP_CONFIG_PATH}"
    echo "[Gripper] 텔레옵 스테이션 USE_GRIPPER = $1"
}

_set_gripper_gene() {
    ssh -t gene@${GENE_WIFI_IP} "
        sed -i 's/^USE_GRIPPER = .*/USE_GRIPPER = $1/' ${GENE_CONFIG_PATH}
        echo '[Gripper] gene config.py USE_GRIPPER = $1'
    "
}

xrgripperon() {
    _set_gripper_local True
    _set_gripper_gene True
    echo "[Gripper] ON — 다음 xrneck / 텔레옵 재시작부터 반영됩니다"
}

xrgripperoff() {
    _set_gripper_local False
    _set_gripper_gene False
    echo "[Gripper] OFF — 다음 xrneck / 텔레옵 재시작부터 반영됩니다"
}

xrgazebo_gene() {
    if [[ "$ROS_MASTER_URI" != *"${GENE_WIFI_IP}"* ]]; then
        echo "⚠️  실제 로봇 모드에서만 사용 가능. 'rmr' 먼저 실행하세요."
        return 1
    fi
    echo "[Gazebo] Gene PC에서 Gazebo 시작..."
    ssh -t gene "
        export ROS_MASTER_URI=http://192.168.1.1:11311
        export ROS_IP=192.168.1.1
        source /opt/ros/noetic/setup.bash
        source ~/catkin_ws/devel/setup.bash
        roslaunch Wholebody_39_DoF_URDF gazebo.launch gui:=false
    "
}

xrhomepose() {
    conda deactivate 2>/dev/null
    local durations="${1:-3.0,3.0,3.0,3.0}"
    rosrun trajectory_manager home_pose_init.py _segment_durations:="${durations}"
}

alias xrcamerastop='fuser -k 8080/tcp 2>/dev/null; \
    ssh ${CURRENT_JETSON_SSH} "pkill -f realsense; pkill -f nodelet" 2>/dev/null; \
    echo "[Camera] 정지 완료"'

alias xrcalib='rm -f "$CONTROL_DIR/calib.json" && echo "[Calib] calib.json 삭제됨"'

alias xrrecord='roslaunch trajectory_manager record.launch   topic:=/joint_group_position_controller/command'

xrplayer() {
    local traj_dir="$HOME/trajectories"

    shopt -s nullglob
    local files=("$traj_dir"/*.yaml)
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then
        echo "No YAML files found in $traj_dir"
        return 1
    fi

    echo "Select a trajectory file to play:"
    select filepath in "${files[@]}"; do
        if [ -n "$filepath" ]; then
            echo -e "\nRunning: roslaunch trajectory_manager play.launch trajectory_file:=$filepath"
            roslaunch trajectory_manager play.launch trajectory_file:="$filepath"
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done
}

alias actcameratestrun='roslaunch realsense2_camera rs_camera.launch \
  color_width:=640 color_height:=480 color_fps:=30 enable_depth:=false'
alias actcameraview='conda deactivate && python3 -c "
import rospy
from sensor_msgs.msg import Image
import cv2, numpy as np

def cb(msg):
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    cv2.imshow(\"camera\", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

rospy.init_node(\"viewer\", anonymous=True)
rospy.Subscriber(\"/dataset_player/camera\", Image, cb)
rospy.spin()
"'
actrecord() {
    conda deactivate
    local task="${1:-humanoid_cup}"
    roslaunch trajectory_manager record_dataset.launch task_name:="${task}"
}
actplaytest() {
    conda deactivate
    local dataset_dir="$HOME/datasets"

    # 1. 하위 디렉토리 목록 가져오기
    shopt -s nullglob
    local dirs=("$dataset_dir"/*/)
    shopt -u nullglob

    if [ ${#dirs[@]} -eq 0 ]; then
        echo "No datasets folders found in $dataset_dir"
        return 1
    fi

    echo "=== Step 1: Choose Datasets Folder ==="
    select selected_dir in "${dirs[@]}"; do
        if [ -n "$selected_dir" ]; then
            echo -e "\n선택된 폴더: $selected_dir"
            break
        else
            echo "올바른 폴더 번호를 입력해주세요."
        fi
    done

    # 사용자가 중간에 취소(Ctrl+D)했는지 체크
    if [ -z "$selected_dir" ]; then
        return 1
    fi

    # 2. 선택된 폴더 안의 hdf5 파일 목록 가져오기
    shopt -s nullglob
    local files=("${selected_dir}"*.hdf5)
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then
        echo "선택한 폴더에 hdf5 파일이 없습니다."
        return 1
    fi

    echo -e "\n=== Step 2: Choose Episode ==="
    select filepath in "${files[@]}"; do
        if [ -n "$filepath" ]; then
            echo -e "\nRunning: roslaunch trajectory_manager play_dataset.launch episode:=$filepath"
            roslaunch trajectory_manager play_dataset.launch episode:="$filepath"
            break
        else
            echo "올바른 파일 번호를 입력해주세요."
        fi
    done
}


xrhelp() {
    local CYAN='\033[0;36m'
    local GREEN='\033[0;32m'
    local YELLOW='\033[1;33m'
    local BLUE='\033[0;34m'
    local PURPLE='\033[0;35m'
    local ORANGE='\033[0;33m'
    local BOLD='\033[1m'
    local NC='\033[0m'

    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${BOLD}XR TELEOP SYSTEM COMMANDS${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"

    echo -e "  ${YELLOW}[1. ROS 모드 전환]${NC}"
    printf "  ${GREEN}%-20s${NC} %s\n" "rme" "실험 모드 (Station = Master)"
    printf "  ${GREEN}%-20s${NC} %s\n" "rmr" "실제 로봇 모드 (Gene PC = Master)"
    printf "  ${GREEN}%-20s${NC} %s\n" "rms" "현재 ROS 네트워크 상태 확인"
    echo ""

    echo -e "  ${YELLOW}[2. 공통 명령어]${NC}"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrteleop" "텔레오퍼레이션 실행"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrcamerastart" "Jetson 카메라 시작"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrcamerastop" "Jetson 카메라 정지"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrcalib" "팔 캘리브레이션 초기화 (calib.json 삭제)"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrhandarduinogene" "Gene PC Arduino Hand Control 실행"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrhandarduinojetson" "Jetson Arduino Hand Control 실행"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrneck" "Gene Dynamixel Neck Node 실행"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrgripperon" "그리퍼 사용 ON (station + gene config)"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrgripperoff" "그리퍼 사용 OFF (station + gene config)"
    printf "  ${BLUE}%-20s${NC} %s\n" "xrhomepose" "홈 포즈로 초기화"
    echo ""

    echo -e "  ${YELLOW}[3. 실험 모드 전용]${NC}"
    printf "  ${PURPLE}%-20s${NC} %s\n" "xrgazebo" "Gazebo 시뮬레이션 실행"
    printf "  ${PURPLE}%-20s${NC} %s\n" "xrrviz_gazebo" "RViz 실시간 로봇 시각화"
    echo ""

    echo -e "  ${YELLOW}[4. 실제 로봇 모드 전용]${NC}"
    printf "  ${PURPLE}%-20s${NC} %s\n" "xrgazebo_gene" "Gene PC에서 Gazebo 실행"
    printf "  ${PURPLE}%-20s${NC} %s\n" "xrrviz_real" "RViz 실시간 로봇 시각화"
    echo ""

    echo -e "  ${YELLOW}[5. 데이터 수집 / ACT]${NC}"
    printf "  ${ORANGE}%-20s${NC} %s\n" "actrecord" "데이터셋 녹화 (task_name 지정)"
    printf "  ${ORANGE}%-20s${NC} %s\n" "actcameraview" "카메라 뷰어 (dataset_player 토픽 구독)"
    printf "  ${ORANGE}%-20s${NC} %s\n" "actplaytest" "녹화된 데이터셋 재생 테스트"
    printf "  ${ORANGE}%-20s${NC} %s\n" "xrrecord" "Teleoperation Position 녹화"
    printf "  ${ORANGE}%-20s${NC} %s\n" "xrplayer" "Teleoperation Position 재생"
    echo ""

    echo -e "  ${YELLOW}[6. SSH 접속]${NC}"
    printf "  ${GREEN}%-20s${NC} %s\n" "jetson" "Jetson 접속"
    printf "  ${GREEN}%-20s${NC} %s\n" "gene" "Gene PC 접속"

    echo -e "${CYAN}──────────────────────────────────────────────────────────────${NC}"
    echo -e "  ${BOLD}🔗 ACCESS LINKS${NC}"
    echo -e "  ${BLUE}Quest  :${NC} https://localhost:8012?ws=wss://localhost:8012"
    echo -e "  ${BLUE}Stream :${NC} http://localhost:8080/stream?topic=/camera/color/image_raw&type=mjpeg"
    echo ""
    echo -e "  ${BOLD}📋 실험 모드 실행 순서:${NC}"
    echo -e "  xrgazebo ➔ xrteleop"
    echo ""
    echo -e "  ${BOLD}📋 실제 로봇 모드 실행 순서:${NC}"
    echo -e "  (Gene PC: roscore) ➔ rmr ➔ xrcamerastart ➔ xrneck & xrhandarduinojetson ➔ xrteleop"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
}
