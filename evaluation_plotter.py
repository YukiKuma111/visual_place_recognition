import rospy
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import math
import threading


class PosePlotter:
    def __init__(self, pose_topic, opt_pose_topic, recommend_thread=1.0):
        self.pose_topic = pose_topic
        self.opt_pose_topic = opt_pose_topic
        self.x_coords = []
        self.y_coords = []
        self.opt_x_coords = []
        self.opt_y_coords = []
        self.total_count = 0
        self.invalid_count = 0
        self.recommend_thread = recommend_thread

        # 初始化 Matplotlib 图形
        self.fig, self.ax = plt.subplots()

        # 蓝色轨迹（底层，粗）
        (self.line,) = self.ax.plot(
            [],
            [],
            marker="o",
            linestyle="-",
            color="b",
            linewidth=2,
            label="Trajectory",
        )

        # 红色轨迹（顶层，细，虚线，空心圆）
        (self.opt_line,) = self.ax.plot(
            [],
            [],
            marker="o",
            linestyle="--",
            color="r",
            markerfacecolor="none",
            linewidth=0.5,
            label="Optimized Trajectory",
        )

        self.ax.set_title("Robot Trajectory")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.legend()
        self.ax.grid()

        # 添加锁来避免多线程冲突
        self.lock = threading.Lock()

    def pose_callback(self, msg):
        # 提取 x 和 y 坐标
        x = msg.pose.position.x
        y = msg.pose.position.y
        time = msg.header.stamp

        with self.lock:
            self.x_coords.append(x)
            self.y_coords.append(y)

            # 增加评估逻辑
            self.total_count += 1
            if len(self.x_coords) > 1:
                # 计算当前点与前一帧的距离
                prev_x = self.x_coords[-2]
                prev_y = self.y_coords[-2]
                distance = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if distance > self.recommend_thread:
                    self.invalid_count += 1

    def opt_pose_callback(self, msg):
        # 提取优化后的 x 和 y 坐标
        x = msg.pose.position.x
        y = msg.pose.position.y

        with self.lock:
            self.opt_x_coords.append(x)
            self.opt_y_coords.append(y)

    def update_plot(self):
        # 更新图形
        with self.lock:
            self.line.set_data(self.x_coords, self.y_coords)
            self.opt_line.set_data(self.opt_x_coords, self.opt_y_coords)
            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)  # 确保 GUI 不会卡住

    def print_evaluation(self):
        if self.total_count > 0:
            invalid_percentage = (self.invalid_count / self.total_count) * 100
            rospy.loginfo(
                f"Total frames: {self.total_count}, Invalid frames: {self.invalid_count}, Invalid percentage: {invalid_percentage:.2f}%"
            )
        else:
            rospy.loginfo("No frames received yet.")

    def start(self):
        rospy.Subscriber(
            self.pose_topic, PoseStamped, self.pose_callback, queue_size=1000
        )
        rospy.Subscriber(
            self.opt_pose_topic, PoseStamped, self.opt_pose_callback, queue_size=1000
        )
        rospy.Timer(
            rospy.Duration(10.0), lambda event: self.print_evaluation()
        )  # 每 10 秒打印一次评估结果
        rospy.loginfo(f"Subscribed to {self.pose_topic} and {self.opt_pose_topic}")

        # 开启 Matplotlib 交互模式
        plt.ion()
        plt.show()

        # 在主线程循环中定期刷新图像
        rate = rospy.Rate(20)  # 20 Hz 刷新频率
        while not rospy.is_shutdown():
            self.update_plot()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("pose_plotter_node")

    # 初始化 PosePlotter
    plotter = PosePlotter(
        pose_topic="/matched_pose",
        opt_pose_topic="/opt_match_pose",
        recommend_thread=1.0,
    )

    # 开始订阅和绘图
    try:
        plotter.start()
    except rospy.ROSInterruptException:
        rospy.loginfo("Pose plotter node terminated.")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
