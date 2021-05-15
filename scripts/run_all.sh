#!/bin/bash

FIRMWARE_PATH="~/PX4-Autopilot"
INNOSIM_PATH="~/dozer/sim"
QGC_PATH="~/QGC"


START_LAT="55.7667439"
START_LON="48.7401153"
START_ALT="0"

# inno
# START_LAT="55.7544426"
# START_LON="48.742684"
# START_ALT="-6.5"


tmux start-server

sleep 1

tmux new -s innosim -d
tmux rename-window -t innosim innosim


tmux split-window -v -t innosim

tmux select-pane -t innosim:0.0
tmux split-window -h -t innosim
tmux split-window -h -t innosim

tmux select-pane -t innosim:0.3
tmux split-window -h -t innosim
tmux split-window -h -t innosim


tmux select-pane -t innosim:0.2
tmux send-keys "cd $FIRMWARE_PATH
export PX4_HOME_LAT=$START_LAT
export PX4_HOME_LON=$START_LON
export PX4_HOME_ALT=$START_ALT
make px4_sitl gazebo" C-m

tmux select-pane -t innosim:0.3

tmux send-keys "roscd inno_sim_interface/cfg
$INNOSIM_PATH/construction_sim.x86_64" C-m

tmux select-pane -t innosim:0.4
tmux send-keys "$QGC_PATH/QGroundControl.AppImage" C-m

sleep 1

tmux select-pane -t innosim:0.0
tmux send-keys 'roslaunch inno_sim_interface innosim_relay.launch' C-m

sleep 1

tmux select-pane -t innosim:0.1
tmux send-keys "roslaunch rosbridge_server rosbridge_websocket.launch" C-m


tmux select-pane -t innosim:0.5

tmux attach -t innosim
