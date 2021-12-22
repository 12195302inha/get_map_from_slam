import math
import random
import rclpy
import threading
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

show_animation = True


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            # 1. 랜덤 노드 생성
            rnd_node = self.get_random_node()

            # 2. 현재 트리에서 랜덤 노드에 가장 가까운 노드 선택
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # 3. 선택된 노드에서 랜덤 노드 방향으로 expand_dis만큼 떨어진 곳에 새로운 노드 생성
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 4. 새로 만든 노드가 밖에 있는지 and 충돌하는지 체크 후 노드에 추가함.
            if self.check_if_outside_play_area(new_node, self.play_area) and \
                    self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            # 5. 목표 지점을 찾았는지 확인
            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        # 만약 delta 거리가 d보다 작다면 맞춰줌.
        if extend_length > d:
            extend_length = d

        # 목표치까지의 거리를 resol로 나눈 정수값을 구함
        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):  # 위에서 정해준 개수만큼
            new_node.x += self.path_resolution * math.cos(theta)  # x,y의 좌표들을
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)  # path에 추가함.
            new_node.path_y.append(new_node.y)

        # 위에서 floor로 내린만큼 추가로 더해줌.
        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        # 새로운 노드를 편입
        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        # 95% 확률로 실수 생성
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        # 5% 확률로 goal point sampling
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        # 절댓값 기준으로 계산하기 위해 제곱
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
                node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= size ** 2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        # 거리 및 theta를 구함.
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)  # 유클리드 크기 반환 (피타고라스)
        theta = math.atan2(dy, dx)  # 탄젠트
        return d, theta


class PlanPublisher(Node):

    def __init__(self):
        # plan setter
        super().__init__('rrt_plan')
        self.plan_publisher = self.create_publisher(Path, "rrt_plan", 1)
        self.timer = self.create_timer(1, self.publish_plan)
        self.map_instance = GetMapFromSlam.instance()

        # rrt planner
        obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                        (9, 5, 2), (8, 10, 1)]
        self.rrt = RRT(
            start=[self.map_instance.pose_x, self.map_instance.pose_y],
            goal=[-2.95, -2.56],
            rand_area=[-2, 15],
            obstacle_list=obstacleList
        )
        path = self.rrt.planning()
        if path is None:
            print("Cannot find path")
        else:
            print(path)
            print("found path!!")

    def publish_plan(self):
        path = Path()
        path.header.frame_id = 'map'

        for now_pose in [[0.0, 0.0], [0.5, 0.25], [1.0, 0.5], [1.5, 0.75]]:
            pose = PoseStamped()
            pose.pose.position.x = now_pose[0]
            pose.pose.position.y = now_pose[1]
            path.poses.append(pose)

        self.plan_publisher.publish(path)


class GetMapFromSlam(Node):
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kwargs):
        cls.__instance = cls(*args, **kwargs)
        cls.instance = cls.__getInstance
        return cls.__instance

    def __init__(self):
        self.occupancy_map = []
        self.map_width = 0.0
        self.map_height = 0.0
        self.resolution = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_z = 0.0
        self.orientation_x = 0.0
        self.orientation_y = 0.0
        self.orientation_z = 0.0
        self.orientation_w = 0.0

        super().__init__('get_map_from_slam')
        qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            qos_profile
        )

    def map_callback(self, msg):
        self.occupancy_map = msg.data  # 100 = occupancy, 0 = free, -1 = unfounded
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.resolution = msg.info.resolution
        self.pose_x = msg.info.origin.position.x
        self.pose_y = msg.info.origin.position.y
        self.pose_z = msg.info.origin.position.z
        self.orientation_x = msg.info.origin.orientation.x
        self.orientation_y = msg.info.origin.orientation.y
        self.orientation_z = msg.info.origin.orientation.z
        self.orientation_w = msg.info.origin.orientation.w
        for i in range(self.map_height):
            print(self.occupancy_map[i * self.map_width:i * self.map_width + self.map_width])


def main(args=None):
    rclpy.init(args=args)
    get_map_node = GetMapFromSlam.instance()
    plan_publisher = PlanPublisher()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_map_node)
    executor.add_node(plan_publisher)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    try:
        while rclpy.ok():
            pass
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        executor_thread.join()


if __name__ == '__main__':
    main()
