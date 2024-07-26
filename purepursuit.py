import numpy as np
import math
import matplotlib.pyplot as plt
import rospy
import pandas as pd
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
from std_msgs.msg import Float64
from tf.transformations import quaternion_from_euler, euler_from_quaternion

# Constants
WB = 2.5  # [m] Wheelbase
dt = 0.1  # [s] Time tick

# Helper functions
def proportional_control(target, current):
    a = 1.0 * (target - current)
    return a

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        self.vx = 0.0
        self.vy = 0.0

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        self.vx = self.v * math.cos(self.yaw)
        self.vy = self.v * math.sin(self.yaw)

class TargetCourse:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = self.calc_distance(state, ind)
            while True:
                distance_next_index = self.calc_distance(state, ind + 1)
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = 0.1 * state.v + 2.0  # look-ahead distance
        while Lf > self.calc_distance(state, ind):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1

        return ind, Lf

    def calc_distance(self, state, point_index):
        dx = state.x - self.cx[point_index]
        dy = state.y - self.cy[point_index]
        return np.hypot(dx, dy)

def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)
    quaternion = quaternion_from_euler(0, 0, delta)

    return delta, quaternion, ind

def main():
    file_path = '~/Desktop/car-racing/local_coordinates_tags.csv'
    df = pd.read_csv(file_path)
    lat_list = list(df['local_x'])
    lon_list = list(df['local_y'])
    cx = []
    cy = []
    for i in lat_list:
        cx.append(i)
    for j in lon_list:
        cy.append(j)
    target_speed = 10.0 / 3.6  # [m/s]

    T = 100.0  # max simulation timeg

    # Initial state
    state = State(x=-0.0, y=0.0, yaw=0.0, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = []
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)
    rospy.init_node('target_vehicle')
    target_publisher = rospy.Publisher('/target_pose', Pose, queue_size=2)
    target_velocity_publisher = rospy.Publisher('/target_velocity', Float64, queue_size=2)
    target_angular_publisher = rospy.Publisher('/target_angular', Float64,queue_size=2 )
    while T >= time and lastIndex > target_ind:
        # Calc control input
        ai = proportional_control(target_speed, state.v)
        delta, quaternion, target_ind = pure_pursuit_steer_control(state, target_course, target_ind)

        state.update(ai, delta)  # Control vehicle

        time += dt
        states.append([state.x, state.y, state.yaw])

        target_pose = Pose()
        target_pose.position.x = state.x
        target_pose.position.y = state.y
        target_pose.position.z = 0
        target_pose.orientation.x = quaternion[0]
        target_pose.orientation.y = quaternion[1]
        target_pose.orientation.z = quaternion[2]
        target_pose.orientation.w = quaternion[3]
        target_publisher.publish(target_pose)
        target_velocity_publisher.publish(state.v)
        target_angular_publisher.publish(state.v / WB * math.tan(delta) * dt)


        rospy.loginfo(f"Velocity in x direction (vx): {state.vx}")
        rospy.loginfo(f"Velocity in y direction (vy): {state.vy}")

        if True:  # show animation
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course", linewidth=0.1)
            plt.plot([x[0] for x in states], [x[1] for x in states], "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot reach goal"

    if True:  # show animation
        plt.cla()
        plt.plot(cx, cy, ".r", label="course", linewidth=0.1)
        plt.plot([x[0] for x in states], [x[1] for x in states], "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(np.arange(0, len(states)) * dt, [state[2] * 3.6 for state in states], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()
