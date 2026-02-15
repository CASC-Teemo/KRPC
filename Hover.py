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
switch = False  # 动力着陆横向导引模式切换标志
dir_mode = True  # 稳定指向模式标志
gear_flag = False  # 起落架展开标志
ref_throttle = 1  # 动力着陆节流阀基准值
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


def output(reentry=False, err_h=0, err_p=0, err_r=0, err_v=0, terminal=False, err_e=0, err_i=0, landing=False,
           err_lateral=0, err_yaw=0, err_speed=0):
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
        print('偏航角误差：%.1f' % err_h, '°')
        print('俯仰角误差：%.1f' % err_p, '°')
        print('滚转角误差：%.1f' % err_r, '°')
        print('垂直角偏差：%.1f' % err_v, '°')
    if terminal:
        print('速度方向误差：%.1f' % err_e, '°')
        print('俯仰角速度误差：%.1f' % err_i, '°')
    if landing:
        print('横向误差：%.1f' % err_lateral, 'm')
        print('目标航向：%.1f' % err_yaw, '°')
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


# 箭体垂直时俯仰角，垂直向上为90，偏东逐渐减小
def pitch_angle():
    v_up = (0, 1, 0)  # 本体系箭体指向
    v_up_t = conn.space_center.transform_direction(v_up, vessel_frame, target_frame)  # 使用着陆点坐标系表示箭体指向
    angle = math.acos(limit(v_up_t[2] / (math.sqrt(v_up_t[0] ** 2 + v_up_t[2] ** 2)), -1, 1)) / math.pi * 180
    return angle


# 箭体垂直时偏航角，垂直向上时为0，偏南(本体系向右)为正
def yaw_angle():
    v_up = (0, 1, 0)  # 本体系箭体指向
    v_up_t = conn.space_center.transform_direction(v_up, vessel_frame, target_frame)  # 使用着陆点坐标系表示箭体指向
    angle = math.acos(limit(v_up_t[1] / (math.sqrt(v_up_t[0] ** 2 + v_up_t[1] ** 2)), -1, 1)) / math.pi * 180
    return angle - 90


# 箭体垂直时滚转角，本体系下方向东为90(基准)，向北为0
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


ctrl.throttle = ref_throttle
throttle_pid = PID(kp=0.1, ki=0.1, kd=0.2, integral_limit=1)
east_speed_pid = PID(kp=2, ki=0, kd=0)
north_speed_pid = PID(kp=2, ki=0, kd=0)
pitch_pid = PID(kp=0.05, ki=0, kd=0.15)
yaw_pid = PID(kp=0.05, ki=0, kd=0.2)
roll_pid = PID(kp=-0.01, ki=0, kd=-0.02)
ut = conn.space_center.ut
target_pos = (1, 0, 0)
target_pitch = 90
target_yaw = 0
target_roll = 90
while True:
    sleep(0.02)
    dt = conn.space_center.ut - ut
    if dt < 0.005:
        continue
    ut = conn.space_center.ut
    target_east_speed = limit((target_pos[2] - vessel.position(target_frame)[2]) / 2, -5, 5)
    err_east_speed = target_frame_velocity()[2] - target_east_speed
    target_pitch = limit(east_speed_pid.update(err_east_speed, dt), -15, 15) + 90
    err_pitch = target_pitch - pitch_angle()
    ctrl.pitch = pitch_pid.update(err_pitch, dt)
    target_north_speed = limit((target_pos[1] - vessel.position(target_frame)[1]) / 2, -5, 5)
    err_north_speed = target_frame_velocity()[1] - target_north_speed
    target_yaw = limit(east_speed_pid.update(err_north_speed, dt), -15, 15)
    err_yaw = target_yaw - yaw_angle()
    ctrl.yaw = yaw_pid.update(err_yaw, dt)
    err_roll = target_roll - roll_angle()
    ctrl.roll = roll_pid.update(err_roll, dt)
    err_v = target_pos[0] - vessel.position(target_frame)[0]
    ctrl.throttle = limit(throttle_pid.update(err_v, dt), -3, 3)
    output(True, err_yaw, err_pitch, err_roll)
