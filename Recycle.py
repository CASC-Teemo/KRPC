import krpc
import math
import numba as nb
from numpy import *
from os import system
from time import sleep

conn = krpc.connect(name='KSP')  # 建立RPC通信
vessel = conn.space_center.active_vessel  # 激活航天器控制
print('连接成功！')
print('任务名：', vessel.name)
ctrl = vessel.control
t_up_v = (0, 1, 0)
g = vessel.orbit.body.surface_gravity  # 海平面重力加速度
srf_frame = vessel.orbit.body.reference_frame  # 质心固连参考系
vessel_frame = vessel.reference_frame  # 本体参考系，设箭体朝向为前，xyz对应右前下
R = vessel.orbit.body.equatorial_radius  # 地球半径
turn_start_speed = 50  # 程序转弯起始速度
turn_end_speed = 450  # 程序转弯结束速度
end_angle = 75  # 转向结束角度
target_altitude = 81000  # 目标轨道高度
terminal = False  # 末制导开始标志
terminal_dist = 5000  # 末制导起始距离
start_angle = -1  # 用于记录开始末制导时速度向量与目标向量的夹角
factor = 1.1  # 着陆点火高度因子，调小以更早进行点火，调大反之
landing = False  # 动力着陆开始标志
gear_flag = False  # 起落架展开标志
ref_throttle = 0.85  # 动力着陆节流阀基准值
target_latitude = 19.614791  # 着陆点纬度
target_longitude = 110.949463  # 着陆点经度
target_height = vessel.orbit.body.surface_height(target_latitude,
                                                 target_longitude) + vessel.orbit.body.equatorial_radius + 7.5  # 着陆点高度
# 构建着陆点东北天坐标系
temp1 = conn.space_center.ReferenceFrame.create_relative(srf_frame, rotation=(
    0, sin(-target_longitude / 2 * math.pi / 180), 0, cos(-target_longitude / 2 * pi / 180)))
temp2 = conn.space_center.ReferenceFrame.create_relative(temp1, rotation=(
    0, 0, sin(target_latitude / 2 * pi / 180), cos(target_latitude / 2 * pi / 180)))
target_frame = conn.space_center.ReferenceFrame.create_relative(temp2, position=(target_height, 0, 0))


# 调试输出
def output(reentry=False, err_h=0, err_p=0, err_r=0, err_v=0, terminal=False, err_e=0, err_i=0, landing=False,
           err_lateral=0, err_heading=0, err_speed=0):
    system('cls')
    print('远点高度：%.1f' % vessel.orbit.apoapsis_altitude, 'm')
    print('当前高度：%.1f' % vessel.position(target_frame)[0], 'm')
    print('水平距离：%.1f' % math.sqrt(vessel.position(target_frame)[1] ** 2 + vessel.position(target_frame)[2] ** 2),
          'm')
    print('地面速度：%.1f' % vessel.flight(srf_frame).speed, 'm/s')
    print('航向角：%.1f' % vessel.flight(target_frame).heading, '°')
    print('俯仰角：%.1f' % vessel.flight(target_frame).pitch, '°')
    print('滚转角：%.1f' % vessel.flight(target_frame).roll, '°')
    if reentry:
        print('俯仰角速度：%.1f' % (-vessel.angular_velocity(srf_frame)[1] * 180 / math.pi), '°')
    else:
        print('俯仰角速度：%.1f' % (vessel.angular_velocity(srf_frame)[1] * 180 / math.pi), '°')
    print('攻角：%.1f' % aoa(), '°')
    if reentry:
        print()
        print('航向角误差：%.1f' % err_h, '°')
        print('俯仰角误差：%.1f' % err_p, '°')
        print('滚转角误差：%.1f' % err_r, '°')
        print('垂直角偏差：%.1f' % err_v, '°')
    if terminal:
        print('速度方向误差：%.1f' % err_e, '°')
        print('俯仰角速度误差：%.1f' % err_i, '°')
    if landing:
        print('横向误差：%.1f' % err_lateral, 'm')
        print('目标航向：%.1f' % err_heading, '°')
        print('速度误差：%.1f' % err_speed, 'm/s')


# 限幅函数
@nb.jit()
def limit(num, min, max):
    if min > max:
        temp = max
        max = min
        min = temp
    if num < min:
        return min
    elif num > max:
        return max
    else:
        return num


# 矢量叉积
@nb.jit()
def cross_product(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


# 矢量点积
@nb.jit()
def dot_product(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


# 矢量模长
@nb.jit()
def magnitude(v):
    return math.sqrt(dot_product(v, v))


# 矢量夹角
@nb.jit()
def vectors_angle(u, v):
    dp = dot_product(u, v)
    if dp == 0:
        return math.pi / 2 * (180 / math.pi)
    um = magnitude(u)
    vm = magnitude(v)
    return math.acos(dp / (um * vm)) * (180 / math.pi)


# 箭体垂直时俯仰角，垂直向上为90°，偏东逐渐减小
def pitch_angle():
    t_up = (1, 0, 0)  # 着陆点坐标系天顶方向
    t_up_v = conn.space_center.transform_direction(t_up, target_frame, vessel_frame)  # 使用本体系表示着陆点坐标系天顶方向
    v_right = (1, 0, 0)  # 本体系右方
    v_down = (0, 0, 1)  # 本体系下方
    pitch_hor = cross_product(v_right, t_up_v)
    angle = vectors_angle(pitch_hor, v_down)
    if pitch_hor[1] < 0:
        return angle + 90
    else:
        return -angle + 90


# 箭体垂直时偏航角，垂直向上时为0，偏南(本体系向右)为正
def yaw_angle():
    t_up = (1, 0, 0)  # 着陆点坐标系天顶方向
    v_forward = (0, 1, 0)  # 本体系前方
    t_up_v = conn.space_center.transform_direction(t_up, target_frame, vessel_frame)
    proj_xoy = (t_up_v[0], t_up_v[1], 0)
    angle = vectors_angle(proj_xoy, v_forward)
    cross = cross_product(v_forward, proj_xoy)
    if cross[2] < 0:
        return -angle
    else:
        return angle


# 箭体垂直时滚转角，本体系下方向东为90°(基准)，向北为0°
def roll_angle():
    t_east = (0, 0, 1)
    v_right = (1, 0, 0)
    v_right_t = conn.space_center.transform_direction(v_right, vessel_frame, target_frame)
    proj_horizontal = (0, v_right_t[1], v_right_t[2])
    angle = vectors_angle(proj_horizontal, t_east)
    cross = cross_product(t_east, proj_horizontal)
    if cross[0] > 0:
        return angle
    else:
        return -angle


# 二维矢量旋转
def rotate_vector(v, angle):
    cos_a = math.cos(angle / 180 * math.pi)
    sin_a = math.sin(angle / 180 * math.pi)
    x = v[0] * cos_a - v[1] * sin_a
    y = v[0] * sin_a + v[1] * cos_a
    return x, y


# 本体系二维速度矢量，以引擎指向作参考，右前为正
def vessel_vel_2D():
    vector = rotate_vector((target_frame_velocity()[1], target_frame_velocity()[2]),
                           90 - vessel.flight(target_frame).heading)
    return vector[0], -vector[1]


# 攻角
def aoa():
    d = vessel.direction(srf_frame)
    v = vessel.velocity(srf_frame)
    dp = dot_product(d, v)
    vm = magnitude(v)
    angle = abs(math.acos(dp / vm) * (180 / math.pi))
    if dp < 0:
        angle = 180 - angle
    return angle


# 经纬度距离计算
@nb.jit()
def dis(lat1, lon1, lat2, lon2):
    lat1 = lat1 * math.pi / 180
    lon1 = lon1 * math.pi / 180
    lat2 = lat2 * math.pi / 180
    lon2 = lon2 * math.pi / 180
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2  # 使用Haversine公式计算距离
    distance = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return distance


# 计算速度方向与目标方向的垂直偏差角
@nb.jit()
def vertical_angle(pos, vel):
    pos_h = math.sqrt(pos[1] ** 2 + pos[2] ** 2)
    target_angle = math.atan(-pos[0] / pos_h) * 180 / math.pi
    vel_h = math.sqrt(vel[1] ** 2 + vel[2] ** 2)
    vel_angle = math.atan(vel[0] / vel_h) * 180 / math.pi
    d_angle = vel_angle - target_angle
    return d_angle


# 着陆点参考系速度矢量
def target_frame_velocity():
    return vessel.velocity(target_frame)[0], vessel.velocity(target_frame)[1], vessel.velocity(target_frame)[2] + 27


# 抗积分饱和PID
class PID:
    def __init__(self, kp, ki, kd, integral_limit=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.err_prev = 0
        self.integral = 0

    def update(self, err, dt):
        p = err * self.kp
        self.integral += err * dt * self.ki
        self.integral = limit(self.integral, -self.integral_limit, self.integral_limit)
        i = self.integral
        d = (err - self.err_prev) / dt * self.kd
        self.err_prev = err
        return p + i + d

    def reset(self):
        self.err_prev = 0
        self.integral = 0


sleep(3)
for i in range(5):
    system('cls')
    print('发射倒计时：')
    print(5 - i)
    sleep(1)
print('点火！')
vessel.auto_pilot.target_pitch_and_heading(90, 90)  # 设定俯仰角及航向角
vessel.auto_pilot.target_roll = 0  # 设定横滚角
vessel.auto_pilot.engage()  # 开启自动驾驶仪
ctrl.throttle = 1  # 节流阀：100%
ctrl.activate_next_stage()  # 芯一级发动机点火

# 主动段
sleep(1)
while True:
    sleep(0.1)
    output()
    if turn_start_speed < vessel.flight(srf_frame).speed < turn_end_speed:  # 程序转弯
        turn_angle = (vessel.flight(srf_frame).speed - turn_start_speed) / (turn_end_speed - turn_start_speed) * (
                90 - end_angle)
        vessel.auto_pilot.target_pitch_and_heading(90 - turn_angle, 90)
    elif vessel.orbit.apoapsis_altitude > target_altitude:  # 发动机一次关机
        ctrl.throttle = 0
        ctrl.toggle_action_group(2)  # 关闭中心引擎
        ctrl.rcs = True  # 开启反应控制系统
        break

# 返场阶段
sleep(1)
vessel.auto_pilot.target_pitch_and_heading(180, 90)  # 航向调整
while True:  # 分离头锥
    sleep(0.1)
    output()
    if 270 - vessel.flight(target_frame).heading < 1 and vessel.flight(target_frame).pitch < 45:
        ctrl.activate_next_stage()
        break
while True:  # 返场点火
    sleep(0.1)
    output()
    if vessel.flight(target_frame).pitch < 1:
        sleep(1)
        ctrl.throttle = 1
        break
while True:  # 返场制导
    sleep(0.1)
    output()
    v = vessel.flight(srf_frame).vertical_speed
    t = (v + math.sqrt(v * v + 2 * g * vessel.flight().mean_altitude)) / g  # 自由落体至地面所需时间
    predict_dis = vessel.flight(srf_frame).horizontal_speed * t  # 距离预测
    if target_frame_velocity()[2] > 0:  # 航天器向东运动
        predict_dis = - predict_dis  # 向东为负，向西为正
    if dis(vessel.flight().latitude, vessel.flight().longitude, target_latitude, target_longitude) < (
            16000 + predict_dis):  # 预测着陆点距目标点小于16km发动机二次关机
        ctrl.throttle = 0
        break

# 再入阶段
sleep(1)
vessel.auto_pilot.target_pitch_and_heading(65, 90)  # 调整为再入姿态
while True:
    sleep(0.1)
    output()
    if vessel.flight(target_frame).heading - 90 < 1 and vessel.flight(target_frame).pitch - 65 < 1:
        sleep(1)
        break
ctrl.toggle_action_group(1)  # 打开栅格舵
while vessel.position(target_frame)[0] > 60000:
    sleep(0.1)
    output()
ctrl.rcs = False  # 关闭反应控制系统
vessel.auto_pilot.disengage()  # 关闭自动驾驶仪
heading_pid = PID(kp=-0.1, ki=0, kd=-0.05)
pitch_pid = PID(kp=-0.2, ki=-0.02, kd=-0.1)
roll_pid = PID(kp=-0.02, ki=0, kd=-0.01)
target_angle_pid = PID(kp=-0.2, ki=-0.05, kd=-0.2)  # 末制导法向导引外环PID，控制速度方向
pitch_rate_pid = PID(kp=0.2, ki=0.02, kd=0)  # 末制导法向导引内环PID，控制俯仰角速度
ut = conn.space_center.ut  # 时刻
target_heading = 90
K = 1.5  # 航向偏置比例
while True:  # 再入制导
    sleep(0.02)
    dt = conn.space_center.ut - ut  # 时间间隔
    if dt < 0.005:
        continue  # 物理帧未刷新跳过
    ut = conn.space_center.ut
    D = math.sqrt(vessel.position(target_frame)[1] ** 2 + vessel.position(target_frame)[2] ** 2)  # 水平距离
    horizontal_speed = math.sqrt(target_frame_velocity()[1] ** 2 + target_frame_velocity()[2] ** 2)  # 水平速度
    retrograde = math.acos(limit(-target_frame_velocity()[1] / horizontal_speed, -1, 1)) / math.pi * 180  # 速度反向
    acc = vessel.available_thrust / vessel.mass  # 航天器当前最大加速度
    pos = (
        vessel.position(target_frame)[0],
        vessel.position(target_frame)[1] * vessel.position(target_frame)[1] / D,
        vessel.position(target_frame)[2] * vessel.position(target_frame)[2] / D)  # 着陆点
    target_dir = math.acos((vessel.position(target_frame)[1] - 30) / D) * 180 / math.pi  # 目标方向
    target_heading = target_dir + (target_dir - 90) * K  # 航向偏置导引
    if D > 2000:
        target_roll = (90 - vessel.flight(target_frame).heading) * K
    else:  # 水平距离过小时锁定滚转
        target_roll = 0
    err_heading = vessel.flight(target_frame).heading - target_heading
    err_roll = vessel.flight(target_frame).roll - target_roll
    ctrl.yaw = limit(heading_pid.update(err_heading, dt), -0.5, 0.5)
    ctrl.roll = limit(roll_pid.update(err_roll, dt), -0.25, 0.25)
    v_angle = vertical_angle(pos, target_frame_velocity())  # 目标矢量与速度矢量的夹角
    if not terminal:
        target_pitch = vessel.flight(target_frame).pitch - (20 - aoa())  # 固定攻角导引
        err_pitch = vessel.flight(target_frame).pitch - target_pitch
        ctrl.pitch = pitch_pid.update(err_pitch, dt)
        output(True, err_heading, err_pitch, err_roll, v_angle)
        if D < terminal_dist:
            terminal = True
    elif start_angle == -1:
        start_angle = v_angle
        err_pitch = 0
        err_e = 0
        err_i = 0
    else:  # 末制导
        err_e = v_angle - start_angle * D / terminal_dist - 1
        err_i = (-vessel.angular_velocity(target_frame)[1] * 180 / math.pi) - limit(target_angle_pid.update(err_e, dt),
                                                                                    -5, 5)
        ctrl.pitch = pitch_rate_pid.update(err_i, dt)  # 末端法向导引
        igni_h = vessel.flight(srf_frame).speed ** 2 / (2 * (acc * factor - g))  # 预计点火高度
        igni_angle = math.asin(limit(horizontal_speed ** 2 / (2 * D * math.cos(
            (target_dir - retrograde) / 180 * math.pi) * 0.75 * acc), -1, 1)) / math.pi * 180  # 预计点火角度
        output(True, err_heading, err_pitch, err_roll, v_angle, terminal, err_e, err_i)
        if vessel.position(target_frame)[0] < igni_h or igni_angle > 45:
            break

# 动力着陆阶段
landing = True
ctrl.throttle = ref_throttle
speed_pid = PID(kp=0.1, ki=0, kd=0)  # 垂直速度PID
heading_pid = PID(kp=-0.05, ki=0, kd=-0.02)
yaw_correct_pid = PID(kp=0.5, ki=0, kd=0.2)
yaw_pid = PID(kp=0.03, ki=0, kd=0.01)
pitch_pid = PID(kp=-0.05, ki=-0.01, kd=-0.03)
roll_pid = PID(kp=-0.005, ki=0, kd=-0.015)
ut = conn.space_center.ut
target_roll = 90
err_lateral = 0
cur_pos = vessel.position(target_frame)  # 当前位置
while target_frame_velocity()[0] < -0.5 and cur_pos[0] > 0.5:  # 动力着陆制导
    pre_pos = cur_pos
    sleep(0.02)
    dt = conn.space_center.ut - ut
    if dt < 0.005:
        continue
    ut = conn.space_center.ut
    cur_pos = vessel.position(target_frame)
    H = cur_pos[0]  # 当前高度
    D = math.sqrt(cur_pos[1] ** 2 + cur_pos[2] ** 2)  # 水平距离
    ds = math.sqrt((pre_pos[1] - cur_pos[1]) ** 2 + (pre_pos[2] - cur_pos[2]) ** 2)  # 与上一时刻水平位移
    horizontal_speed = math.sqrt(target_frame_velocity()[1] ** 2 + target_frame_velocity()[2] ** 2)  # 水平速度
    retrograde = math.acos(limit((pre_pos[1] - cur_pos[1]) / ds, -1, 1)) / math.pi * 180  # 速度反向
    pos_dir = math.acos(limit(cur_pos[1] / D, -1, 1)) / math.pi * 180  # 目标位置航向
    acc = vessel.available_thrust / vessel.mass
    err_lateral = math.sin((retrograde - pos_dir) / 180 * math.pi) * D  # 预测横向误差
    if conn.space_center.transform_direction((0, 100, 0), vessel_frame, target_frame)[2] < 0:  # 重定义俯仰角
        pitch = 180 - vessel.flight(target_frame).pitch
    else:
        pitch = vessel.flight(target_frame).pitch
    if not gear_flag and H < 350:  # 展开着陆腿
        gear_flag = True
        ctrl.gear = True
    err_roll = target_roll - roll_angle()
    ctrl.roll = limit(roll_pid.update(err_roll, dt), -0.2, 0.2)
    if vessel.position(target_frame)[0] > 30 and target_frame_velocity()[2] < -1:
        if vessel.position(target_frame)[2] > 20:  # 纵向小于20m锁定俯仰角
            target_pitch = 90 - math.asin(limit(horizontal_speed ** 2 / (2 * D * math.cos(
                (pos_dir - retrograde) / 180 * math.pi) * 0.75 * acc), -1, 1)) / math.pi * 180
        target_heading = retrograde
        if vessel.flight(srf_frame).speed < 200:
            target_yaw = limit(yaw_correct_pid.update(err_lateral, dt), -10, 10)  # 预测校正制导
            err_yaw = target_yaw - yaw_angle()
            ctrl.yaw = yaw_pid.update(err_yaw, dt)
        else:
            err_heading = vessel.flight(target_frame).heading - target_heading
            ctrl.yaw = heading_pid.update(err_heading, dt)
    else:  # 接地前修正
        if abs(vessel_vel_2D()[1]) < 1:
            target_pitch = 90
        else:
            target_pitch = 90 - limit(vessel_vel_2D()[1] * 2, -5, 5)
        if abs(vessel_vel_2D()[0]) < 1:
            target_yaw = 0
        else:
            target_yaw = limit(vessel_vel_2D()[0] * 2, -5, 5)
        err_yaw = target_yaw - yaw_angle()
        ctrl.yaw = yaw_pid.update(err_yaw, dt)
    err_pitch = pitch - target_pitch
    ctrl.pitch = pitch_pid.update(err_pitch, dt)
    target_speed = math.sqrt(H * 2 * (acc * ref_throttle - g))  # 垂直速度控制
    err_speed = -target_frame_velocity()[0] - target_speed + 1
    ctrl.throttle = limit(speed_pid.update(err_speed, dt) + ref_throttle, 0.75, 1)
    output(True, err_heading, err_pitch, 0, 0, terminal, 0, 0, landing, err_lateral, target_heading, err_speed)
ctrl.throttle = 0
ctrl.pitch = 0
ctrl.yaw = 0
ctrl.toggle_action_group(3)
ctrl.toggle_action_group(4)
ctrl.rcs = True
ctrl.sas = True
ctrl.sas_mode = ctrl.sas_mode.stability_assist
sleep(3)
ctrl.rcs = False
ctrl.sas = False
system('pause')
