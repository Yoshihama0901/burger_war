#!/bin/bash

kill `ps auxww | grep judgeServer.py | grep -v grep| awk '{print $2}'`
kill `ps auxww | grep visualizeWindow.py | grep -v grep| awk '{print $2}'`

gnome-terminal -e "python ../catkin_ws/src/burger_war/judge/judgeServer.py"
gnome-terminal -e "python ../catkin_ws/src/burger_war/judge/visualizeWindow.py"

rosnode kill /blue_bot/send_id_to_judge
rosnode kill /red_bot/send_id_to_judge

bash ../catkin_ws/src/burger_war/judge/test_scripts/init_single_play.sh ../catkin_ws/src/burger_war/judge/marker_set/sim.csv localhost:5000 you enemy

rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState -- 'red_bot' '[[-0.00802941365317, -1.30180708272, -0.00100240389136], [-0.00289870472068, 0.00255000374612, 0.749537215393, 0.661950948132]]' '[[-2.14486099572e-05, -6.03070056406e-06, 3.57530023394e-05], [-0.000172449078833, -5.47956803684e-05, 0.000337038447954]]' 'world'


rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState -- 'blue_bot' '[[0.00803057422089, 1.30182140949, -0.00100240392709], [0.00254760493418, 0.00290081362037, -0.661330587132, 0.750084628234]]' ' [[2.14379168728e-05, 6.0664025918e-06, 3.57531181111e-05],  [-0.000172449078833, -5.47956803684e-05, 0.000337038447954]]' 'world'


bash ../catkin_ws/src/burger_war/judge/test_scripts/set_running.sh localhost:5000
