#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tf
import csv
import json
import rospy
import random
import subprocess
import numpy as np
import sys
import datetime

from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Vector3, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import Image
import actionlib # RESPECT @seigot
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal # RESPECT @seigot
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rosparam
 
# 強化学習DQN (Deep Q Network)
from MyModule import DQN

#timeScale  = 4    # １秒間で何回座標計算するか？
timeScale  = 4    # １秒間で何回座標計算するか？
fieldScale = 1.5  # 競技場の広さ
#turnEnd    = 40   # 何ターンで１試合を終了させるか
turnEnd    = 10   # 何ターンで１試合を終了させるか


# クォータニオンからオイラー角への変換
def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0]*180/np.pi, y=e[1]*180/np.pi, z=e[2]*180/np.pi)


# 座標回転行列を返す
def get_rotation_matrix(rad):
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return rot


# 現在地を２次元ベクトル(n*n)にして返す
def get_pos_matrix(x, y, n=16):
    #my_pos  = np.array([self.pos[0], self.pos[1]])           # 現在地点
    pos     = np.array([x, y])                                # 現在地点
    rot     = get_rotation_matrix(-45 * np.pi / 180)          # 45度回転行列の定義
    #rotated = ( np.dot(rot, pos) / fieldScale ) + 0.5         # 45度回転して最大幅1.5で正規化(0-1)
    rotated = ( np.dot(rot, pos) + 1 ) / 2                    # 回転を行って0-1の範囲にシフト
    pos_np  = np.zeros([n, n])
    i = int(rotated[0]*n)
    j = int(rotated[1]*n)
    if i < 0: i = 0
    if i > 15: i = 15
    if j < 0: j = 0
    if j > 15: j = 15
    pos_np[i][j] = 1
    return pos_np


# 自分が向いている向きを２次元ベクトル(n*n)にして返す
def get_ang_matrix(angle, n=16):
    while angle > 0 : angle -= 360
    while angle < 0 : angle += 360
    my_ang  = np.zeros([n, n])
    for i in range(16):
        for j in range(16):
            if 360-22.5 < angle or angle <= 22.5 :              #   0°
                if 10 <= i and 10 <= j      : my_ang[i][j] = 1
            if  45-22.5 < angle <=  45+22.5 :                   #  45°
                if 10 <= i and  5 <= j <= 10: my_ang[i][j] = 1
            if  90-22.5 < angle <=  90+22.5 :                   #  90°
                if 10 <= i and  5 >= j      : my_ang[i][j] = 1
            if 135-22.5 < angle <= 135+22.5 :                   # 135°
                if  5 <= i <=10 and  5 >= j : my_ang[i][j] = 1
            if 180-22.5 < angle <= 180+22.5 :                   # 180°
                if  5 >= i and  5 >= j      : my_ang[i][j] = 1
            if 225-22.5 < angle <= 225+22.5 :                   # 225°
                if  5 >= i and  5 <= j <= 10: my_ang[i][j] = 1
            if 270-22.5 < angle <= 270+22.5 :                   # 270°
                if  5 >= i and  10 <= j     : my_ang[i][j] = 1
            if 315-22.5 < angle <= 315+22.5 :                   # 315°
                if  5 <= i <=10 and 10 <= j : my_ang[i][j] = 1
    #print(my_ang)
    return my_ang


# 得点ベクトルを返す
def get_sco_matrix(score, point):
    #point = 1
    np_sco = np.zeros([16, 16])
    if score[8]  == point : np_sco[12,  7] = 1   #  8:Tomato_N
    if score[9]  == point : np_sco[11,  8] = 1   #  9:Tomato_S
    if score[10] == point : np_sco[ 8,  3] = 1   # 10:Omelette_N
    if score[11] == point : np_sco[ 7,  4] = 1   # 11:Omelette_S
    if score[12] == point : np_sco[ 8, 11] = 1   # 12:Pudding_N
    if score[13] == point : np_sco[ 7, 12] = 1   # 13:Pudding_S
    if score[14] == point : np_sco[ 3,  8] = 1   # 14:OctopusWiener_N
    if score[15] == point : np_sco[ 4,  7] = 1   # 15:OctopusWiener_S
    if score[16] == point : np_sco[ 8,  7] = 1   # 16:FriedShrimp_N
    if score[17] == point : np_sco[ 8,  8] = 1   # 17:FriedShrimp_E
    if score[18] == point : np_sco[ 7,  7] = 1   # 18:FriedShrimp_W
    if score[19] == point : np_sco[ 7,  8] = 1   # 19:FriedShrimp_S
    return np_sco

# 自分の側面得点
def get_side_matrix(side1, side2):
    np_sco = np.zeros([16, 16])
    for i in range(16):
        for j in range(16):
            if not side1 == 0 :
                if 7 >= i : np_sco[i][j] = 1
            if not side2 == 0 :
                if 8 <= i : np_sco[i][j] = 1
    return np_sco

# gazebo座標からamcl_pose座標に変換する
def convert_coord_from_gazebo_to_amcl(my_color, gazebo_x, gazebo_y):
    if my_color == 'r':
        amcl_x    =  gazebo_y
        amcl_y    = -gazebo_x
    else:
        amcl_x    = -gazebo_y
        amcl_y    =  gazebo_x
    return amcl_x, amcl_y

class RandomBot():

    # 現在の状態を取得
    def getState(self):
        
        # 位置情報
        my_angle = quaternion_to_euler(Quaternion(self.pos[2], self.pos[3], self.pos[4], self.pos[5]))
        my_pos = get_pos_matrix(self.pos[0], self.pos[1])                      # 自分の位置
        en_pos = get_pos_matrix(self.pos[6], self.pos[7]                    )  # 相手の位置
        my_ang = get_ang_matrix(my_angle.z)                                    # 自分の向き
        
        # 審判情報の更新(点数)
        rospy.Subscriber("war_state", String, self.callback_war_state, queue_size=10)
        my_sco      = get_sco_matrix(self.score,  1)                           # 自分の点数
        en_sco      = get_sco_matrix(self.score, -1)                           # 相手の点数
        mySide_sco  = get_side_matrix(self.score[6], self.score[7])            # 自分側面の点数
        enSide_sco  = get_side_matrix(self.score[3], self.score[4])            # 相手側面の点数

        # 状態と報酬の更新( 16 × 16 × 7ch )
        state       = np.concatenate([np.expand_dims(my_pos,     axis=2),
                                     np.expand_dims(en_pos,     axis=2),
                                     np.expand_dims(my_ang,     axis=2),
                                     np.expand_dims(my_sco,     axis=2),
                                     np.expand_dims(en_sco,     axis=2),
                                     np.expand_dims(mySide_sco, axis=2),
                                     np.expand_dims(enSide_sco, axis=2)], axis=2)
        state       = np.reshape(state, [1, 16, 16, 7])                         # 現在の状態(自分と相手の位置、点数)
        
        return state

    # クラス生成時に最初に呼ばれる
    def __init__(self, bot_name, color='r', Sim_flag=True):
        self.name     = bot_name                                        # bot name 
        self.vel_pub  = rospy.Publisher('cmd_vel', Twist, queue_size=1) # velocity publisher
        self.sta_pub  = rospy.Publisher("/gazebo/model_states", ModelStates, latch=True) # 初期化用
        self.timer    = 0                                               # 対戦時間
        self.reward   = 0.0                                             # 報酬
        self.my_color = color                                           # 自分の色情報
        self.en_color = 'b' if color=='r' else 'r'                      # 相手の色情報
        self.score    = np.zeros(20)                                    # スコア情報(以下詳細)
        self.sim_flag = Sim_flag
         #  0:自分のスコア, 1:相手のスコア
         #  2:相手後ろ, 3:相手Ｌ, 4:相手Ｒ, 5:自分後ろ, 6:自分Ｌ, 7:自分Ｒ
         #  8:Tomato_N, 9:Tomato_S, 10:Omelette_N, 11:Omelette_S, 12:Pudding_N, 13:Pudding_S
         # 14:OctopusWiener_N, 15:OctopusWiener_S, 16:FriedShrimp_N, 17:FriedShrimp_E, 18:FriedShrimp_W, 19:FriedShrimp_S
        self.pos      = np.zeros(12)                                    # 位置情報(以下詳細)
         #  0:自分位置_x,  1:自分位置_y,  2:自分角度_x,  3:自分角度_y,  4:自分角度_z,  5:自分角度_w
         #  6:相手位置_x,  7:相手位置_y,  8:相手角度_x,  9:相手角度_y, 10:相手角度_z, 11:相手角度_w
        self.w_name = "imageview-" + self.my_color
        # cv2.namedWindow(self.w_name, cv2.WINDOW_NORMAL)
        # cv2.moveWindow(self.w_name, 100, 100)
        camera_resource_name = 'image_raw' if self.my_color == 'r' else 'image_raw'
        self.image_pub = rospy.Publisher(camera_resource_name, Image, queue_size=10)
        self.img = None
        self.debug_preview = False
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(camera_resource_name, Image, self.imageCallback, queue_size=10)
        self.debug_log_fname = None
        #self.debug_log_fname = 'log-' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '-' + self.my_color + '.csv'
        self.training = True and self.sim_flag
        self.debug_use_gazebo_my_pos = False
        self.debug_use_gazebo_enemy_pos = False
        self.debug_gazebo_my_x = np.nan
        self.debug_gazebo_my_y = np.nan
        self.debug_gazebo_enemy_x = np.nan
        self.debug_gazebo_enemy_y = np.nan
        if self.debug_use_gazebo_my_pos is False:
            if self.my_color == 'r' : rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.callback_amcl_pose)
            if self.my_color == 'b' : rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.callback_amcl_pose)
        if self.debug_use_gazebo_enemy_pos is False:
            self.pos[6] = 1.3 if self.my_color == 'r' else -1.3
            self.pos[7] = 0
        if (self.debug_use_gazebo_my_pos is True) or (self.debug_use_gazebo_enemy_pos is True) or (self.debug_log_fname is not None):
            rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state, queue_size=10)
        if self.debug_log_fname is not None:
            with open(self.debug_log_fname, mode='a') as f:
                f.write('my_x,my_y,my_qx,my_qy,my_qz,my_qw,my_ax,my_ay,my_az,enemy_x,enemy_y,enemy_qx,enemy_qy,enemy_qz,enemy_qw,enemy_ax,enemy_ay,enemy_az,circle_x,circle_y,circle_r,est_enemy_x,est_enemy_y,est_enemy_u,est_enemy_v,est_enemy_theta,gazebo_my_x,gazebo_my_y,gazebo_enemy_x,gazebo_enemy_y,diff_my_x,diff_my_y,diff_enemy_x,diff_enemy_y\n')
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction) # RESPECT @seigot]
        
        # 初期状態を取得
        #self.state = np.zeros([1, 16, 16, 7])                        # 状態
        self.state = self.getState()
        
        self.action = np.array([0, 0])
        self.action2 = np.array([0, 0])

    # スコア情報の更新(war_stateのコールバック関数)
    def callback_war_state(self, data):
        json_dict = json.loads(data.data)                  # json辞書型に変換
        self.score[0] = json_dict['scores'][self.my_color] # 自分のスコア
        self.score[1] = json_dict['scores'][self.en_color] # 相手のスコア
        if json_dict['state'] == 'running':
            try:
                for i in range(18):
                    #print('*********', len(json_dict['targets']))
                    player = json_dict['targets'][i]['player']
                    if player == self.my_color : self.score[2+i] =  float(json_dict['targets'][i]['point'])
                    if player == self.en_color : self.score[2+i] = -float(json_dict['targets'][i]['point'])
                if self.my_color == 'b':                           # 自分が青色だった場合、相手と自分を入れ替える
                    for i in range(3) : self.score[2+i], self.score[5+i] = self.score[5+i], self.score[2+i]
            except:
                print('callback_war_state: Invalid input ' + e)

    # 位置情報の更新(amcl_poseのコールバック関数)
    def callback_amcl_pose(self, data):
        pos = data.pose.pose.position
        ori = data.pose.pose.orientation
        self.pos[0] = pos.x; self.pos[1] = pos.y; self.pos[2] = ori.x; self.pos[3] = ori.y; self.pos[4] = ori.z; self.pos[5] = ori.w
    
    # 位置情報の更新(model_stateのコールバック関数)
    def callback_model_state(self, data):
        #print('*********', len(data.pose))
        if 'red_bot' in data.name:
            index_r = data.name.index('red_bot')
        else:
            print('callback_model_state: red_bot not found')
            return
        if 'blue_bot' in data.name:
            index_b = data.name.index('blue_bot')
        else:
            print('callback_model_state: blue_bot not found')
            return
        #print('callback_model_state: index_r=', index_r, 'index_b=', index_b)
        my    = index_r if self.my_color == 'r' else index_b
        enemy = index_b if self.my_color == 'r' else index_r
        gazebo_my_x,    gazebo_my_y    = convert_coord_from_gazebo_to_amcl(self.my_color, data.pose[my   ].position.x, data.pose[my   ].position.y)
        gazebo_enemy_x, gazebo_enemy_y = convert_coord_from_gazebo_to_amcl(self.my_color, data.pose[enemy].position.x, data.pose[enemy].position.y)
        if self.debug_use_gazebo_my_pos is True:
            self.pos[0] = gazebo_my_x
            self.pos[1] = gazebo_my_y
            ori = data.pose[my].orientation; self.pos[2] = ori.x; self.pos[3] = ori.y; self.pos[4]  = ori.z; self.pos[5]  = ori.w
        if self.debug_use_gazebo_enemy_pos is True:
            self.pos[6] = gazebo_enemy_x
            self.pos[7] = gazebo_enemy_y
            ori = data.pose[enemy].orientation; self.pos[8] = ori.x; self.pos[9] = ori.y; self.pos[10] = ori.z; self.pos[11] = ori.w
        self.debug_gazebo_my_x    = gazebo_my_x
        self.debug_gazebo_my_y    = gazebo_my_y
        self.debug_gazebo_enemy_x = gazebo_enemy_x
        self.debug_gazebo_enemy_y = gazebo_enemy_y

    # 報酬の計算
    def calc_reward(self):
        
        reward = 0.0
        
        # 部分点
        #reward = ( self.score[0] - self.score[1] ) / 10.0
        #if reward >  1: reward =  1
        #if reward < -1: reward = -1
        
        # 試合終了
        #print('+++***+++', self.score)
        if self.timer > turnEnd:
            if self.score[0] >  self.score[1] : reward =  1
            if self.score[0] <= self.score[1] : reward = -1
        if self.score[0] >= 100 : reward =  1      # 一本勝ち
        if self.score[1] >= 100 : reward = -1      # 一本負け
        if not self.score[2] == 0 : reward =  1    # 一本勝ち
        if not self.score[5] == 0 : reward = -1    # 一本負け
        
        return reward


    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ 行動計算のメイン部
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def calcTwist(self):
        
        self.timer += 1
        
        # 行動を決定する
        #action, linear, angle = self.actor.get_action(self.state, 1, self.mainQN)
        action = self.actor.get_action(self.state, self.timer, self.mainQN, self.my_color, self.action, self.action2, self.score[0]-self.score[1], self.sim_flag)
        if self.timer == 1:
            action = np.array([5, 11])
            self.action2 = self.action
            self.action = action
        
        # 移動先と角度  (中心位置をずらした後に45度反時計周りに回転)
        #pos     = (action - 8) * fieldScale/8                                   # 目的地
        pos     = (action - 8) / 8.0                                            # 目的地
        rot     = get_rotation_matrix(45 * np.pi / 180)                         # 45度回転行列の定義
        desti   = np.dot(rot, pos)                                              # 45度回転
        yaw = np.arctan2( (desti[1]-self.pos[1]), (desti[0]-self.pos[0]) )      # 移動先の角度
        if self.my_color == 'r' :
            #print('****Action****', self.timer, action, desti, yaw*360/np.pi)
            print('*** Action *** Time=%2d,  Position=(%4.2f, %4.2f),  Destination=(%4.2f, %4.2f, %4.0f[deg])' % (self.timer, self.pos[0], self.pos[1], desti[0], desti[1], yaw*360/np.pi))
            print('')
        
        # Actionに従った行動  目的地の設定 (X, Y, Yaw)
        self.setGoal(desti[0], desti[1], yaw)
        #self.restart()  # ******* 強制Restart用 *******
        
        # Action後の状態と報酬を取得
        next_state = self.getState()                                            # Action後の状態
        reward     =  self.calc_reward()                                        # Actionの結果の報酬
        
        # メモリの更新する
        self.memory.add((self.state, action, reward, next_state))               # メモリの更新する
        if abs(reward) == 1 : np.zeros([1, 16, 16, 7])                          # 試合終了時は次の状態はない
        self.state  = next_state                                                 # 状態更新
        self.action2 = self.action
        self.action = action
        
        # Qネットワークの重みを学習・更新する replay
        if self.training == True : learn = 1
        else                     : learn = 0
        if self.my_color == 'b'  : learn = 0
        batch_size = 40   # Q-networkを更新するバッチの大きさ
        #batch_size = self.timer - 1   # Q-networkを更新するバッチの大きさ
        gamma = 0.97      # 割引係数
        if (batch_size >= 2 and self.memory.len() > batch_size) and learn:
            #print('call replay timer=', self.timer)
            self.mainQN.replay(self.memory, batch_size, gamma, self.targetQN, self.my_color)
        self.targetQN.model.set_weights(self.mainQN.model.get_weights())
        
        sys.stdout.flush()
        self.reward = reward
        
        return Twist()
        
        #value = random.randint(1,1000)
        #if   value <  250 : x =  0.2; th =  0
        #elif value <  500 : x = -0.2; th =  0
        #elif value <  750 : x =  0.0; th =  1
        #elif value < 1000 : x =  0.0; th = -1
        #else              : x =  0.0; th =  0
        #twist = Twist()
        #twist.linear.x = x; twist.linear.y = 0; twist.linear.z = 0
        #twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th
        #return twist


    # シュミレーション再開
    def restart(self):
        self.vel_pub.publish(Twist()) # 動きを止める
        self.memory.reset()
        self.score  = np.zeros(20)
        self.timer  = 0
        self.reward = 0
        subprocess.call('bash ../catkin_ws/src/burger_war/burger_war/scripts/reset_state.sh', shell=True)
        #r.sleep()


    # RESPECT @seigot
    # do following command first.
    #   $ roslaunch burger_navigation multi_robot_navigation_run.launch
    #   $ rosservice call move_base_set_logger_level ros.move_base WARN   # 移動時のログを表示しない
    def setGoal(self,x,y,yaw):
        self.client.wait_for_server()
        #print('setGoal x=', x, 'y=', y, 'yaw=', yaw)

        goal = MoveBaseGoal()
        name = 'red_bot' if self.my_color == 'r' else 'blue_bot'
        #goal.target_pose.header.frame_id = name + '/map' if self.sim_flag == True else 'map'
        goal.target_pose.header.frame_id = "map"

        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        # Euler to Quartanion
        q=tf.transformations.quaternion_from_euler(0,0,yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        # Stateの戻り値詳細：PENDING, ACTIVE, RECALLED, REJECTED, PREEMPTED, ABORTED, SUCCEEDED, LOST
        #  https://docs.ros.org/diamondback/api/actionlib/html/classactionlib_1_1SimpleClientGoalState.html#a91066f14351d31404a2179da02c518a0a2f87385336ac64df093b7ea61c76fafe
        #state = self.client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(5))
        state = self.client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(4))
        #print(self.my_color, "state=", state)

        return 0


    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ 戦略部(繰り返し処理を行わせる)
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def strategy(self):
        
        rospy_Rate = timeScale
        r = rospy.Rate(rospy_Rate) # １秒間に送る送信回数 (change speed 1fps)
        
        # Qネットワークとメモリ、Actorの生成--------------------------------------------------------
        learning_rate = 0.0005          # Q-networkの学習係数
        memory_size   = 400             # バッファーメモリの大きさ
        self.mainQN   = DQN.QNetwork(learning_rate=learning_rate)   # メインのQネットワーク
        self.targetQN = DQN.QNetwork(learning_rate=learning_rate)   # 価値を計算するQネットワーク
        self.memory   = DQN.Memory(max_size=memory_size)
        self.actor    = DQN.Actor()
        
        # 重みの読み込み
        if self.sim_flag == True : self.mainQN.model.load_weights('../catkin_ws/src/burger_war/burger_war/scripts/weight.hdf5')     # 重みの読み込み
        else                     : self.mainQN.model.load_weights('../wss/Yoshihama0901_ws/src/burger_war/burger_war/scripts/weight.hdf5')     # 重みの読み込み
        self.targetQN.model.set_weights(self.mainQN.model.get_weights())

        while not rospy.is_shutdown():
            
            twist = self.calcTwist()    # 移動距離と角度を計算
            #self.vel_pub.publish(twist) # ROSに反映
            
            if self.training == True:
                # 試合終了した場合
                if self.my_color == 'r':
                    if abs(self.reward) == 1 or self.timer > turnEnd:
                        if   self.reward == 0 : print('Draw')
                        elif self.reward == 1 : print('Win!')
                        else                  : print('Lose')
                        with open('result.csv', 'a') as f:
                            writer = csv.writer(f, lineterminator='\n')
                            writer.writerow([self.score[0], self.score[1]])
                        self.mainQN.model.save_weights('../catkin_ws/src/burger_war/burger_war/scripts/weight.hdf5')            # モデルの保存
                        self.restart()                                          # 試合再開
                        r.sleep()
                else:
                    if self.timer % turnEnd == 0 :
                        self.memory.reset()
                        self.mainQN.model.load_weights('../catkin_ws/src/burger_war/burger_war/scripts/weight.hdf5')                # 重みの読み込み
            
            r.sleep()
        
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ カメラ画像読込み部分(多分変える事はないと思う)
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def imageCallback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV_FULL)
        hsv_h = hsv[:, :, 0]
        hsv_s = hsv[:, :, 1]
        mask = np.zeros(hsv_h.shape, dtype=np.uint8)
        mask[((hsv_h < 16) | (hsv_h > 240)) & (hsv_s > 64)] = 255
        red = cv2.bitwise_and(self.img, self.img, mask=mask)
        height = self.img.shape[0]
        canny_param = 100
        canny = cv2.Canny(red, canny_param/2, canny_param)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=height/10, param1=canny_param, param2=8,
                                   minRadius=height/96, maxRadius=height/12)
        circle_x = -1
        circle_y = -1
        circle_r = -1
        est_enemy_x = np.nan
        est_enemy_y = np.nan
        est_enemy_u = np.nan
        est_enemy_v = np.nan
        est_enemy_theta = np.nan
        my_x = self.pos[0]
        my_y = self.pos[1]
        my_qx = self.pos[2]
        my_qy = self.pos[3]
        my_qz = self.pos[4]
        my_qw = self.pos[5]
        my_angle = quaternion_to_euler(Quaternion(my_qx, my_qy, my_qz, my_qw))
        if circles is not None:
            for i in circles[0,:]:
                x = int(i[0])
                y = int(i[1])
                r = int(i[2])
                if (y < height * 5 / 8) and (r > circle_r):
                    circle_x = x
                    circle_y = y
                    circle_r = r
            if circle_r > 0:
                est_enemy_sin_theta = -0.00143584 * circle_x \
                                      + 0.4458366274811388
                est_enemy_theta = np.rad2deg(np.arcsin(est_enemy_sin_theta))
                est_enemy_v = 4.58779425e-09 * np.power(circle_y, 4) \
                              - 1.14983273e-06 * np.power(circle_y, 3) \
                              + 1.21335973e-04 * np.power(circle_y, 2) \
                              - 7.94065667e-04 * circle_y \
                              + 0.5704722921109504
                est_enemy_u = -est_enemy_v * np.tan(np.deg2rad(est_enemy_theta))
                est_p = np.cos(np.deg2rad(my_angle.z)) * est_enemy_u \
                        - np.sin(np.deg2rad(my_angle.z)) * est_enemy_v
                est_q = np.sin(np.deg2rad(my_angle.z)) * est_enemy_u \
                        + np.cos(np.deg2rad(my_angle.z)) * est_enemy_v
                est_dx = est_q
                est_dy = -est_p
                est_enemy_x = my_x + est_dx
                est_enemy_y = my_y + est_dy
        if self.debug_use_gazebo_enemy_pos is False:
            if (not np.isnan(est_enemy_x)) and (not np.isnan(est_enemy_y)):
                self.pos[6] = est_enemy_x
                self.pos[7] = est_enemy_y
        if self.debug_log_fname is not None:
            with open(self.debug_log_fname, mode='a') as f:
                # pos[6] ... pos[11] are filled in callback_model_state
                enemy_x = self.pos[6]
                enemy_y = self.pos[7]
                enemy_qx = self.pos[8]
                enemy_qy = self.pos[9]
                enemy_qz = self.pos[10]
                enemy_qw = self.pos[11]
                enemy_angle = quaternion_to_euler(Quaternion(enemy_qx, enemy_qy, enemy_qz, enemy_qw))
                f.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'
                        % (my_x, my_y, my_qx, my_qy, my_qz, my_qw,
                           my_angle.x, my_angle.y, my_angle.z,
                           enemy_x, enemy_y, enemy_qx, enemy_qy, enemy_qz, enemy_qw,
                           enemy_angle.x, enemy_angle.y, enemy_angle.z,
                           circle_x, circle_y, circle_r,
                           est_enemy_x, est_enemy_y, est_enemy_u, est_enemy_v, est_enemy_theta,
                           self.debug_gazebo_my_x, self.debug_gazebo_my_y, self.debug_gazebo_enemy_x, self.debug_gazebo_enemy_y,
                           my_x - self.debug_gazebo_my_x, my_y - self.debug_gazebo_my_y,
                           est_enemy_x - self.debug_gazebo_enemy_x, est_enemy_y - self.debug_gazebo_enemy_y))
        if self.debug_preview:
            hough = self.img.copy()
            if circles is not None:
                for i in circles[0,:]:
                    color = (255, 255, 0)
                    pen_width = 2
                    if circle_x == int(i[0]) and circle_y == int(i[1]):
                        color = (0, 255, 0)
                        pen_width = 4
                        cv2.circle(hough, (int(i[0]), int(i[1])), int(i[2]), color, pen_width)
            #cv2.imshow("red", red)
            #cv2.imshow("canny", canny)
            cv2.imshow(self.w_name, hough)
            cv2.waitKey(1)

if __name__ == '__main__':
    
    # sim環境用のフラグ。本番(実機動作)では、
    #   ・リセット動作を行わない
    #   ・学習を行わない
    #   ・確率でのランダム動作を行わない
    Sim_flag = True
    
    rname = rosparam.get_param('randomRun/rname')
    rside = rosparam.get_param('randomRun/rname')
    if rname == 'red_bot' or rside == 'r': color = 'r'
    else                                 : color = 'b'
    print('****************', rname, rside, color)
    
    rospy.init_node('IntegAI_run')    # 初期化宣言 : このソフトウェアは"IntegAI_run"という名前
    bot = RandomBot('Team Integ AI', color=color, Sim_flag=Sim_flag)
    
    bot.strategy()

