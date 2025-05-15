import math
import heapq
from collections import defaultdict
import random

# Dữ liệu từ file Gridline.xlsx
pickup_points = {
    "18.1": (3, 0), "18.2": (4, 0), "18.3": (5, 0), "18.4": (6, 0),
    "18.5": (7, 0), "18.6": (8, 0), "18.7": (9, 0), "18.8": (10, 0),
    "17.1": (3, 4), "17.2": (4, 4), "17.3": (5, 4), "17.4": (6, 4),
    "17.5": (7, 4), "17.6": (8, 4), "17.7": (9, 4), "17.8": (10, 4),
    "16.1": (3, 8), "16.2": (4, 8), "16.3": (5, 8), "16.4": (6, 8),
    "16.5": (7, 8), "16.6": (8, 8), "16.7": (9, 8), "16.8": (10, 8),
    "15.1": (3, 14.481), "15.2": (4, 14.481), "15.3": (5, 14.481), "15.4": (6, 14.481),
    "15.5": (7, 14.481), "15.6": (8, 14.481), "15.7": (9, 14.481), "15.8": (10, 14.481),
    "14.1": (3, 18.481), "14.2": (4, 18.481), "14.3": (5, 18.481), "14.4": (6, 18.481),
    "14.5": (7, 18.481), "14.6": (8, 18.481), "14.7": (9, 18.481), "14.8": (10, 18.481),
    "13.1": (3, 22.481), "13.2": (4, 22.481), "13.3": (5, 22.481), "13.4": (6, 22.481),
    "13.5": (7, 22.481), "13.6": (8, 22.481), "13.7": (9, 22.481), "13.8": (10, 22.481),
    "12.1": (3, 26.481), "12.2": (4, 26.481), "12.3": (5, 26.481), "12.4": (6, 26.481),
    "12.5": (7, 26.481), "12.6": (8, 26.481), "12.7": (9, 26.481), "12.8": (10, 26.481),
    "11.1": (3, 32.957), "11.2": (4, 32.957), "11.3": (5, 32.957), "11.4": (6, 32.957),
    "11.5": (7, 32.957), "11.6": (8, 32.957), "11.7": (9, 32.957), "11.8": (10, 32.957),
    "10.1": (3, 36.957), "10.2": (4, 36.957), "10.3": (5, 36.957), "10.4": (6, 36.957),
    "10.5": (7, 36.957), "10.6": (8, 36.957), "10.7": (9, 36.957), "10.8": (10, 36.957),
    "9.1": (3, 40.957), "9.2": (4, 40.957), "9.3": (5, 40.957), "9.4": (6, 40.957),
    "9.5": (7, 40.957), "9.6": (8, 40.957), "9.7": (9, 40.957), "9.8": (10, 40.957),
    "8.1": (3, 44.957), "8.2": (4, 44.957), "8.3": (5, 44.957), "8.4": (6, 44.957),
    "8.5": (7, 44.957), "8.6": (8, 44.957), "8.7": (9, 44.957), "8.8": (10, 44.957),
    "7.1": (3, 48.957), "7.2": (4, 48.957), "7.3": (5, 48.957), "7.4": (6, 48.957),
    "7.5": (7, 48.957), "7.6": (8, 48.957), "7.7": (9, 48.957), "7.8": (10, 48.957),
    "6.1": (3, 52.957), "6.2": (4, 52.957), "6.3": (5, 52.957), "6.4": (6, 52.957),
    "6.5": (7, 52.957), "6.6": (8, 52.957), "6.7": (9, 52.957), "6.8": (10, 52.957),
    "5.1": (3, 59.143), "5.2": (4, 59.143), "5.3": (5, 59.143), "5.4": (6, 59.143),
    "5.5": (7, 59.143), "5.6": (8, 59.143), "5.7": (9, 59.143), "5.8": (10, 59.143),
    "4.1": (3, 63.143), "4.2": (4, 63.143), "4.3": (5, 63.143), "4.4": (6, 63.143),
    "4.5": (7, 63.143), "4.6": (8, 63.143), "4.7": (9, 63.143), "4.8": (10, 63.143),
    "3.1": (3, 67.143), "3.2": (4, 67.143), "3.3": (5, 67.143), "3.4": (6, 67.143),
    "3.5": (7, 67.143), "3.6": (8, 67.143), "3.7": (9, 67.143), "3.8": (10, 67.143),
    "2.1": (3, 71.143), "2.2": (4, 71.143), "2.3": (5, 71.143), "2.4": (6, 71.143),
    "2.5": (7, 71.143), "2.6": (8, 71.143), "2.7": (9, 71.143), "2.8": (10, 71.143),
    "1.1": (3, 75.143), "1.2": (4, 75.143), "1.3": (5, 75.143), "1.4": (6, 75.143),
    "1.5": (7, 75.143), "1.6": (8, 75.143), "1.7": (9, 75.143), "1.8": (10, 75.143),
}

vertices = {
    1: (0, 0), 2: (0, 4), 3: (0, 8), 4: (0, 14.481), 5: (0, 18.481), 6: (0, 22.481),
    7: (0, 26.481), 8: (0, 32.957), 9: (0, 36.957), 10: (0, 40.957), 11: (0, 44.957),
    12: (0, 48.957), 13: (0, 52.957), 14: (0, 59.143), 15: (0, 63.143), 16: (0, 67.143),
    17: (0, 71.143), 18: (0, 75.143), 19: (1.5, -0.75), 20: (1.5, 0), 21: (1.5, 2),
    22: (1.5, 4), 23: (1.5, 6), 24: (1.5, 8), 25: (1.5, 8.75), 26: (1.5, 13.731),
    27: (1.5, 14.481), 28: (1.5, 16.481), 29: (1.5, 18.481), 30: (1.5, 20.481),
    31: (1.5, 22.481), 32: (1.5, 24.481), 33: (1.5, 26.481), 34: (1.5, 27.231),
    35: (1.5, 32.207), 36: (1.5, 32.957), 37: (1.5, 34.957), 38: (1.5, 36.957),
    39: (1.5, 38.957), 40: (1.5, 40.957), 41: (1.5, 42.957), 42: (1.5, 44.957),
    43: (1.5, 46.957), 44: (1.5, 48.957), 45: (1.5, 50.957), 46: (1.5, 52.957),
    47: (1.5, 53.707), 48: (1.5, 58.393), 49: (1.5, 59.143), 50: (1.5, 59.893),
    51: (1.5, 61.143), 52: (1.5, 63.143), 53: (1.5, 65.143), 54: (1.5, 67.143),
    55: (1.5, 69.143), 56: (1.5, 71.143), 57: (1.5, 73.143), 58: (1.5, 75.143),
    59: (1.5, 77.143), 110: (14.5, -3), 111: (14.5, -1), 112: (14.5, 0), 113: (14.5, 2),
    114: (14.5, 4), 115: (14.5, 6), 116: (14.5, 8), 117: (14.5, 8.75), 118: (14.5, 13.731),
    119: (14.5, 14.481), 120: (14.5, 16.481), 121: (14.5, 18.481), 122: (14.5, 20.481),
    123: (14.5, 22.481), 124: (14.5, 24.481), 125: (14.5, 26.481), 126: (14.5, 27.231),
    127: (14.5, 32.207), 128: (14.5, 32.957), 129: (14.5, 34.957), 130: (14.5, 36.957),
    131: (14.5, 38.957), 132: (14.5, 40.957), 133: (14.5, 42.957), 134: (14.5, 44.957),
    135: (14.5, 46.957), 136: (14.5, 48.957), 137: (14.5, 50.957), 138: (14.5, 52.957),
    139: (14.5, 53.707), 140: (14.5, 58.393), 141: (14.5, 59.143), 142: (14.5, 59.893),
    143: (14.5, 61.143), 144: (14.5, 63.143), 145: (14.5, 65.143), 146: (14.5, 67.143),
    147: (14.5, 69.143), 148: (14.5, 71.143), 149: (14.5, 73.143), 150: (14.5, 75.143),
    151: (14.5, 77.143), 152: (14.5, 83.6), 153: (14.5, 85.6), 154: (14.5, 91.1),
    155: (14.5, 93.1), 156: (14.5, 98.6), 157: (14.5, 100.6), 158: (14.5, 105.1),
    159: (14.5, 107.1), 160: (18.191, -3), 161: (18.191, -1), 162: (18.191, 14.481),
    163: (18.191, 16.481), 164: (18.191, 18.481), 165: (18.191, 20.481), 166: (18.191, 32.957),
    167: (18.191, 34.957), 168: (18.191, 36.957), 169: (18.191, 38.957), 170: (18.191, 40.957),
    171: (18.191, 42.957), 172: (18.191, 44.957), 173: (18.191, 46.957), 174: (18.191, 48.957),
    175: (18.191, 50.957), 176: (18.191, 52.957), 177: (18.191, 59.893), 178: (18.191, 61.143),
    179: (18.191, 63.143), 180: (18.191, 65.143), 181: (18.191, 75.143), 182: (18.191, 77.143),
    183: (18.191, 83.6), 184: (18.191, 85.6), 185: (18.191, 91.1), 186: (18.191, 93.1),
    187: (18.191, 98.6), 188: (18.191, 100.6), 189: (18.191, 105.1), 190: (18.191, 107.1)
}

# Xây dựng đồ thị
graph = defaultdict(list)
node_coords = vertices.copy()
node_id = len(vertices) + 1

# Thêm pickup points vào đồ thị
for pp, coord in pickup_points.items():
    node_coords[node_id] = coord
    graph[node_id] = []
    node_id += 1

# Ánh xạ pickup points và vertices
node_to_id = {coord: nid for nid, coord in node_coords.items()}

# Tạo cạnh giữa các node (dùng Manhattan distance)
for i in node_coords:
    for j in node_coords:
        if i >= j:
            continue
        xi, yi = node_coords[i]
        xj, yj = node_coords[j]
        if xi == xj or yi == yj:  # Kết nối nếu cùng hàng hoặc cùng cột
            blocked = False
            for k in node_coords:
                if k == i or k == j:
                    continue
                xk, yk = node_coords[k]
                if xi == xj and xk == xi:
                    if min(yi, yj) < yk < max(yi, yj):
                        blocked = True
                        break
                if yi == yj and yk == yi:
                    if min(xi, xj) < xk < max(xi, xj):
                        blocked = True
                        break
            if not blocked:
                dist = abs(xi - xj) + abs(yi - yj)
                graph[i].append((j, dist))
                graph[j].append((i, dist))

# Ánh xạ dock tại X = 0
dock_to_vertex = {
    18: 1, 17: 2, 16: 3, 15: 4, 14: 5, 13: 6, 12: 7, 11: 8, 10: 9,
    9: 10, 8: 11, 7: 12, 6: 13, 5: 14, 4: 15, 3: 16, 2: 17, 1: 18
}

# Conveyor nodes tại X = 18.191
conveyor_nodes = list(range(160, 191))
if not conveyor_nodes:
    raise ValueError("Conveyor nodes list is empty. Check node coordinates for x = 18.191.")

# Lớp Task
class Task:
    def __init__(self, dock, zone, task_id, t_call=0):
        self.dock = dock
        self.zone = zone
        self.task_id = task_id
        self.start_node = dock_to_vertex[dock] if zone == 1 else random.choice(conveyor_nodes)
        self.conveyor_node = random.choice(conveyor_nodes) if zone == 1 else dock_to_vertex[dock]
        dock_y = node_coords[dock_to_vertex[dock]][1]
        possible_pickups = [(k, v) for k, v in pickup_points.items() if v[1] == dock_y]
        self.pickup_node = node_to_id[random.choice(possible_pickups)[1]]
        self.t_call = t_call
        self.t_start = None
        self.t_completion = random.uniform(252, 282) if zone == 1 else random.uniform(120, 175)
        self.wait_time = 0
        self.assigned_forklift = None
        self.phase = 0
        self.next_call_delay = None
        self.completion_time = None

# Lớp Forklift
class Forklift:
    def __init__(self, id, start_node):
        self.id = id
        self.current_node = start_node
        self.t_free = 0
        self.path = []
        self.current_task = None
        self.tasks_completed_per_dock = defaultdict(int)
        self.schedule = []
        self.constraints = []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

# Thuật toán A* với ràng buộc thời gian từ CBS
def a_star_with_constraints(graph, start, goal, constraints, start_time, speed):
    open_set = [(0, start, start_time)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited = set()

    while open_set:
        _, current, current_time = heapq.heappop(open_set)
        if current == goal:
            path = []
            time = current_time
            node = current
            while node in came_from:
                path.append((node, time))
                node, time = came_from[node]
            path.append((start, start_time))
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)

        for neighbor, dist in graph[current]:
            time_to_move = dist / speed
            arrival_time = current_time + time_to_move
            conflict = False
            for constraint in constraints:
                if constraint['type'] == 'vertex' and constraint['node'] == neighbor and abs(constraint['time'] - arrival_time) < 1e-6:
                    conflict = True
                    break
                if constraint['type'] == 'edge' and constraint['from'] == current and constraint['to'] == neighbor and abs(constraint['time'] - current_time) < 1e-6:
                    conflict = True
                    break
            if conflict:
                continue

            tentative_g_score = g_score.get(current, float('inf')) + dist
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = (current, current_time)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor, arrival_time))
    return []

def heuristic(node1, node2):
    x1, y1 = node_coords[node1]
    x2, y2 = node_coords[node2]
    return abs(x1 - x2) + abs(y1 - y2)

# Hàm phát hiện xung đột
def detect_conflicts(forklifts):
    conflicts = []
    for i, forklift1 in enumerate(forklifts):
        if not forklift1.schedule:
            continue
        for j, forklift2 in enumerate(forklifts):
            if i >= j or not forklift2.schedule:
                continue
            for idx1, (node1, time1) in enumerate(forklift1.schedule):
                for idx2, (node2, time2) in enumerate(forklift2.schedule):
                    if node1 == node2 and abs(time1 - time2) < 1e-6:
                        conflicts.append({
                            'type': 'vertex',
                            'node': node1,
                            'time': time1,
                            'forklifts': (forklift1.id, forklift2.id)
                        })
                        break
            for idx1 in range(len(forklift1.schedule) - 1):
                node1, time1 = forklift1.schedule[idx1]
                node2, time2 = forklift1.schedule[idx1 + 1]
                for idx2 in range(len(forklift2.schedule) - 1):
                    node3, time3 = forklift2.schedule[idx2]
                    node4, time4 = forklift2.schedule[idx2 + 1]
                    if node1 == node4 and node2 == node3 and abs(time1 - time3) < 1e-6:
                        conflicts.append({
                            'type': 'edge',
                            'from': node1,
                            'to': node2,
                            'time': time1,
                            'forklifts': (forklift1.id, forklift2.id)
                        })
                        break
    return conflicts

# Mô phỏng
def simulate():
    global_task_id = 0
    dock_tasks = defaultdict(list)
    total_tasks = 0
    docks = list(range(1, 19))
    zone_1_docks = [dock for dock in docks if dock >= 10]
    zone_2_docks = [dock for dock in docks if dock < 10]
    task_counts = {}
    for dock in zone_1_docks:
        task_counts[dock] = random.randint(38, 42)
    for dock in zone_2_docks:
        task_counts[dock] = random.randint(40, 50)
    initial_task_counts = {}
    for dock in docks:
        zone = 1 if dock >= 10 else 2
        num_tasks = task_counts[dock]
        initial_task_counts[dock] = num_tasks
        t_call = 0
        for i in range(num_tasks):
            task = Task(dock, zone, global_task_id, t_call)
            dock_tasks[dock].append(task)
            global_task_id += 1
            total_tasks += 1

    print("Initial Number of Tasks per Dock (Before Simulation):")
    print("Zone 1 (Dock 10-18):")
    print("Dock | Initial Number of Tasks")
    print("-" * 30)
    for dock in sorted(zone_1_docks, reverse=True):
        print(f"{dock:4} | {initial_task_counts[dock]:20}")
    print("\nZone 2 (Dock 1-9):")
    print("Dock | Initial Number of Tasks")
    print("-" * 30)
    for dock in sorted(zone_2_docks):
        print(f"{dock:4} | {initial_task_counts[dock]:20}")

    start_nodes = [19, 20, 21, 22, 23, 24, 25, 26, 27]
    forklifts = [Forklift(i, start_nodes[i]) for i in range(9)]
    speed = 1.42
    dock_wait_times = defaultdict(float)
    dock_task_counts = defaultdict(int)
    task_details = defaultdict(list)
    t = 0
    active_tasks = []
    completed_tasks_count = 0
    max_simulation_time = 10800
    dock_current_task = {dock: [] for dock in docks}
    remaining_tasks = {dock: tasks[:] for dock, tasks in dock_tasks.items()}
    last_task_added_time = {dock: 0 for dock in docks}
    min_task_interval = 150
    max_active_tasks = total_tasks * 2
    conflict_counts = defaultdict(int)  # Đếm số lần xung đột cho mỗi cặp xe nâng
    max_conflict_threshold = 10  # Ngưỡng tối đa cho xung đột lặp lại

    print(f"\nTotal tasks to complete: {total_tasks}")
    while completed_tasks_count < total_tasks and t < max_simulation_time:
        if t % 1000 == 0:
            available_forklifts = len([f for f in forklifts if f.t_free <= t and not f.current_task])
            print(f"Time: {t}, Completed tasks: {completed_tasks_count}, Active tasks: {len(active_tasks)}, Available forklifts: {available_forklifts}")

        # Thêm nhiệm vụ mới
        for dock in dock_current_task:
            max_tasks = 2 if dock in zone_1_docks else 1
            time_since_last_task = t - last_task_added_time[dock]
            can_add_task = dock not in zone_1_docks or time_since_last_task >= min_task_interval
            while len(dock_current_task[dock]) < max_tasks and remaining_tasks[dock] and can_add_task and len(active_tasks) < max_active_tasks:
                task = remaining_tasks[dock][0]
                task.t_call = t
                dock_current_task[dock].append(task)
                remaining_tasks[dock].pop(0)
                last_task_added_time[dock] = t
                print(f"Added task {task.task_id} to dock {dock} at time {t}")

        # Kiểm tra nhiệm vụ hoàn thành tại dock/conveyor
        for dock in dock_current_task:
            completed_tasks = []
            for task in dock_current_task[dock]:
                if t >= task.t_call + task.t_completion and task.phase == 0:
                    task.completion_time = t
                    if task not in active_tasks:
                        active_tasks.append(task)
                        print(f"Task {task.task_id} (Dock {dock}, Zone {task.zone}) moved to active_tasks, phase={task.phase}, completion_time={task.completion_time:.2f}")
                    completed_tasks.append(task)
            dock_current_task[dock] = [task for task in dock_current_task[dock] if task not in completed_tasks]

        # Phân bổ xe nâng
        tasks_to_assign = [task for task in active_tasks if task.assigned_forklift is None and task.phase == 0]
        if tasks_to_assign:
            tasks_to_assign.sort(key=lambda task: t - task.completion_time, reverse=True)
            for task in tasks_to_assign:
                available_forklifts = [f for f in forklifts if f.t_free <= t and not f.current_task]
                if available_forklifts:
                    forklift = min(available_forklifts, key=lambda f: (
                        f.t_free,
                        heuristic(f.current_node, task.start_node) * 2,
                        f.tasks_completed_per_dock[task.dock]
                    ))
                    path = a_star_with_constraints(graph, forklift.current_node, task.start_node, forklift.constraints, forklift.t_free, speed)
                    if not path:
                        print(f"Warning: No path found for forklift {forklift.id} from {forklift.current_node} to {task.start_node}, constraints: {len(forklift.constraints)}")
                        task.t_call = t + 50
                        continue

                    forklift.schedule = path
                    max_cbs_attempts = 10
                    for attempt in range(max_cbs_attempts):
                        conflicts = detect_conflicts(forklifts)
                        if not conflicts:
                            conflict_counts.clear()
                            break
                        conflict = conflicts[0]
                        forklift1_id, forklift2_id = conflict['forklifts']
                        # Xử lý conflict_key dựa trên loại xung đột
                        if conflict['type'] == 'vertex':
                            conflict_key = (min(forklift1_id, forklift2_id), max(forklift1_id, forklift2_id), conflict['node'], conflict['time'])
                        elif conflict['type'] == 'edge':
                            conflict_key = (min(forklift1_id, forklift2_id), max(forklift1_id, forklift2_id), (conflict['from'], conflict['to']), conflict['time'])
                        else:
                            print(f"Warning: Unknown conflict type {conflict['type']} for forklifts {forklift1_id} and {forklift2_id}")
                            task.t_call = t + 50
                            forklift.schedule = []
                            break

                        conflict_counts[conflict_key] += 1
                        print(f"Conflict detected: Forklift {forklift1_id} vs Forklift {forklift2_id}, type={conflict['type']}, time={conflict['time']:.2f}, attempt={attempt+1}, conflict_count={conflict_counts[conflict_key]}")

                        # Kiểm tra xung đột dai dẳng
                        if conflict_counts[conflict_key] > max_conflict_threshold:
                            print(f"Persistent conflict detected for forklifts {forklift1_id} and {forklift2_id} at time {conflict['time']:.2f}. Skipping task {task.task_id}.")
                            task.t_call = t + 50
                            forklift.schedule = []
                            for f in forklifts:
                                f.constraints = []
                            conflict_counts.clear()
                            break

                        forklift1 = next(f for f in forklifts if f.id == forklift1_id)
                        forklift2 = next(f for f in forklifts if f.id == forklift2_id)
                        if conflict['type'] == 'vertex':
                            forklift1.add_constraint({'type': 'vertex', 'node': conflict['node'], 'time': conflict['time']})
                            target1 = task.start_node if forklift1.id == forklift.id else (forklift1.current_task.start_node if forklift1.current_task else task.start_node)
                            path1 = a_star_with_constraints(graph, forklift1.current_node, target1, forklift1.constraints, forklift1.t_free, speed)
                            if path1:
                                forklift1.schedule = path1
                            else:
                                print(f"Warning: Forklift {forklift1.id} failed to find path to {target1}")
                                forklift1.constraints.pop()
                                task.t_call = t + 50
                                break
                            forklift2.add_constraint({'type': 'vertex', 'node': conflict['node'], 'time': conflict['time']})
                            target2 = forklift2.current_task.start_node if forklift2.current_task else task.start_node
                            path2 = a_star_with_constraints(graph, forklift2.current_node, target2, forklift2.constraints, forklift2.t_free, speed)
                            if path2:
                                forklift2.schedule = path2
                            else:
                                print(f"Warning: Forklift {forklift2.id} failed to find path to {target2}")
                                forklift2.constraints.pop()
                                task.t_call = t + 50
                                break
                        elif conflict['type'] == 'edge':
                            forklift1.add_constraint({'type': 'edge', 'from': conflict['from'], 'to': conflict['to'], 'time': conflict['time']})
                            target1 = task.start_node if forklift1.id == forklift.id else (forklift1.current_task.start_node if forklift1.current_task else task.start_node)
                            path1 = a_star_with_constraints(graph, forklift1.current_node, target1, forklift1.constraints, forklift1.t_free, speed)
                            if path1:
                                forklift1.schedule = path1
                            else:
                                print(f"Warning: Forklift {forklift1.id} failed to find path to {target1}")
                                forklift1.constraints.pop()
                                task.t_call = t + 50
                                break
                            forklift2.add_constraint({'type': 'edge', 'from': conflict['to'], 'to': conflict['from'], 'time': conflict['time']})
                            target2 = forklift2.current_task.start_node if forklift2.current_task else task.start_node
                            path2 = a_star_with_constraints(graph, forklift2.current_node, target2, forklift2.constraints, forklift2.t_free, speed)
                            if path2:
                                forklift2.schedule = path2
                            else:
                                print(f"Warning: Forklift {forklift2.id} failed to find path to {target2}")
                                forklift2.constraints.pop()
                                task.t_call = t + 50
                                break
                    else:
                        print(f"Warning: CBS failed for forklift {forklift.id} after {max_cbs_attempts} attempts for task {task.task_id}")
                        task.t_call = t + 50
                        forklift.schedule = []
                        for f in forklifts:
                            f.constraints = []
                        conflict_counts.clear()
                        continue

                    # Kiểm tra xem schedule có rỗng hay không trước khi truy cập
                    if not forklift.schedule:
                        print(f"Warning: Forklift {forklift.id} schedule is empty after CBS for task {task.task_id}")
                        task.t_call = t + 50
                        continue

                    _, arrival_time = forklift.schedule[-1]
                    if arrival_time < t:
                        arrival_time = t
                    if arrival_time == task.completion_time:
                        arrival_time += random.uniform(5, 10)

                    print(f"Forklift {forklift.id} assigned to Zone {task.zone} for task {task.task_id} at Dock {task.dock}, arrival_time={arrival_time:.2f}")
                    task.assigned_forklift = forklift
                    task.t_start = arrival_time
                    task.wait_time = task.t_start - task.completion_time
                    dock_wait_times[task.dock] += task.wait_time
                    dock_task_counts[task.dock] += 1
                    task_details[task.dock].append({
                        "task_id": task.task_id,
                        "completion_time": task.t_completion,
                        "wait_time": task.wait_time,
                        "dock_completion_time": task.completion_time,
                        "forklift_start_time": task.t_start
                    })
                    forklift.current_task = task
                    forklift.t_free = arrival_time
                    forklift.path = [(node, time) for node, time in path[1:]] if len(path) > 1 else []

        # Cập nhật trạng thái xe nâng
        for forklift in forklifts:
            if forklift.current_task and forklift.path:
                node, arrival_time = forklift.path[0]
                if t >= arrival_time:
                    forklift.current_node = node
                    forklift.path.pop(0)
                    forklift.t_free = t
                    forklift.schedule = [(n, time) for n, time in forklift.schedule if time > t]

            if forklift.current_task and not forklift.path:
                task = forklift.current_task
                if task.phase == 0:
                    forklift.t_free = t + 12
                    task.phase = 1
                    path = a_star_with_constraints(graph, forklift.current_node, task.pickup_node, forklift.constraints, forklift.t_free, speed)
                    if not path:
                        print(f"Warning: No path to pickup for task {task.task_id}, resetting to phase 0")
                        task.phase = 0
                        task.t_call = t + 50
                        forklift.current_task = None
                        continue
                    forklift.schedule = path
                    for attempt in range(10):
                        conflicts = detect_conflicts(forklifts)
                        if not conflicts:
                            conflict_counts.clear()
                            break
                        conflict = conflicts[0]
                        forklift1_id, forklift2_id = conflict['forklifts']
                        if conflict['type'] == 'vertex':
                            conflict_key = (min(forklift1_id, forklift2_id), max(forklift1_id, forklift2_id), conflict['node'], conflict['time'])
                        elif conflict['type'] == 'edge':
                            conflict_key = (min(forklift1_id, forklift2_id), max(forklift1_id, forklift2_id), (conflict['from'], conflict['to']), conflict['time'])
                        else:
                            print(f"Warning: Unknown conflict type {conflict['type']} in phase 1 for forklifts {forklift1_id} and {forklift2_id}")
                            task.t_call = t + 50
                            forklift.schedule = []
                            break

                        conflict_counts[conflict_key] += 1
                        print(f"Conflict detected in phase 1: Forklift {forklift1_id} vs Forklift {forklift2_id}, type={conflict['type']}, time={conflict['time']:.2f}, conflict_count={conflict_counts[conflict_key]}")
                        if conflict_counts[conflict_key] > max_conflict_threshold:
                            print(f"Persistent conflict detected in phase 1 for forklifts {forklift1_id} and {forklift2_id}. Resetting task {task.task_id}.")
                            task.t_call = t + 50
                            forklift.schedule = []
                            for f in forklifts:
                                f.constraints = []
                            conflict_counts.clear()
                            break
                        forklift1 = next(f for f in forklifts if f.id == forklift1_id)
                        forklift2 = next(f for f in forklifts if f.id == forklift2_id)
                        if conflict['type'] == 'vertex':
                            forklift1.add_constraint({'type': 'vertex', 'node': conflict['node'], 'time': conflict['time']})
                            target1 = task.pickup_node if forklift1.id == forklift.id else (forklift1.current_task.start_node if forklift1.current_task else task.pickup_node)
                            path1 = a_star_with_constraints(graph, forklift1.current_node, target1, forklift1.constraints, forklift1.t_free, speed)
                            if path1:
                                forklift1.schedule = path1
                            else:
                                print(f"Warning: Forklift {forklift1.id} failed to find path to {target1}")
                                forklift1.constraints.pop()
                                task.t_call = t + 50
                                break
                            forklift2.add_constraint({'type': 'vertex', 'node': conflict['node'], 'time': conflict['time']})
                            target2 = forklift2.current_task.start_node if forklift2.current_task else task.pickup_node
                            path2 = a_star_with_constraints(graph, forklift2.current_node, target2, forklift2.constraints, forklift2.t_free, speed)
                            if path2:
                                forklift2.schedule = path2
                            else:
                                print(f"Warning: Forklift {forklift2.id} failed to find path to {target2}")
                                forklift2.constraints.pop()
                                task.t_call = t + 50
                                break
                        elif conflict['type'] == 'edge':
                            forklift1.add_constraint({'type': 'edge', 'from': conflict['from'], 'to': conflict['to'], 'time': conflict['time']})
                            target1 = task.pickup_node if forklift1.id == forklift.id else (forklift1.current_task.start_node if forklift1.current_task else task.pickup_node)
                            path1 = a_star_with_constraints(graph, forklift1.current_node, target1, forklift1.constraints, forklift1.t_free, speed)
                            if path1:
                                forklift1.schedule = path1
                            else:
                                print(f"Warning: Forklift {forklift1.id} failed to find path to {target1}")
                                forklift1.constraints.pop()
                                task.t_call = t + 50
                                break
                            forklift2.add_constraint({'type': 'edge', 'from': conflict['to'], 'to': conflict['from'], 'time': conflict['time']})
                            target2 = forklift2.current_task.start_node if forklift2.current_task else task.pickup_node
                            path2 = a_star_with_constraints(graph, forklift2.current_node, target2, forklift2.constraints, forklift2.t_free, speed)
                            if path2:
                                forklift2.schedule = path2
                            else:
                                print(f"Warning: Forklift {forklift2.id} failed to find path to {target2}")
                                forklift2.constraints.pop()
                                task.t_call = t + 50
                                break
                    if not forklift.schedule:
                        print(f"Warning: Forklift {forklift.id} schedule is empty after CBS in phase 1 for task {task.task_id}")
                        task.phase = 0
                        task.t_call = t + 50
                        forklift.current_task = None
                        continue
                    forklift.path = [(node, time) for node, time in path[1:]] if len(path) > 1 else []
                    print(f"Task {task.task_id} moved to phase 1, heading to pickup node {task.pickup_node}")
                elif task.phase == 1:
                    pickup_time = random.uniform(10, 13)
                    forklift.t_free = t + pickup_time
                    task.next_call_delay = random.uniform(60, 90)
                    task.t_call = t + pickup_time + task.next_call_delay
                    task.phase = 2
                    goal = task.conveyor_node if task.zone == 1 else task.start_node
                    path = a_star_with_constraints(graph, forklift.current_node, goal, forklift.constraints, forklift.t_free, speed)
                    if not path:
                        print(f"Warning: No path to goal for task {task.task_id}, resetting to phase 1")
                        task.phase = 1
                        task.t_call = t + 50
                        forklift.current_task = None
                        continue
                    forklift.schedule = path
                    for attempt in range(10):
                        conflicts = detect_conflicts(forklifts)
                        if not conflicts:
                            conflict_counts.clear()
                            break
                        conflict = conflicts[0]
                        forklift1_id, forklift2_id = conflict['forklifts']
                        if conflict['type'] == 'vertex':
                            conflict_key = (min(forklift1_id, forklift2_id), max(forklift1_id, forklift2_id), conflict['node'], conflict['time'])
                        elif conflict['type'] == 'edge':
                            conflict_key = (min(forklift1_id, forklift2_id), max(forklift1_id, forklift2_id), (conflict['from'], conflict['to']), conflict['time'])
                        else:
                            print(f"Warning: Unknown conflict type {conflict['type']} in phase 2 for forklifts {forklift1_id} and {forklift2_id}")
                            task.t_call = t + 50
                            forklift.schedule = []
                            break

                        conflict_counts[conflict_key] += 1
                        print(f"Conflict detected in phase 2: Forklift {forklift1_id} vs Forklift {forklift2_id}, type={conflict['type']}, time={conflict['time']:.2f}, conflict_count={conflict_counts[conflict_key]}")
                        if conflict_counts[conflict_key] > max_conflict_threshold:
                            print(f"Persistent conflict detected in phase 2 for forklifts {forklift1_id} and {forklift2_id}. Resetting task {task.task_id}.")
                            task.t_call = t + 50
                            forklift.schedule = []
                            for f in forklifts:
                                f.constraints = []
                            conflict_counts.clear()
                            break
                        forklift1 = next(f for f in forklifts if f.id == forklift1_id)
                        forklift2 = next(f for f in forklifts if f.id == forklift2_id)
                        if conflict['type'] == 'vertex':
                            forklift1.add_constraint({'type': 'vertex', 'node': conflict['node'], 'time': conflict['time']})
                            target1 = goal if forklift1.id == forklift.id else (forklift1.current_task.start_node if forklift1.current_task else goal)
                            path1 = a_star_with_constraints(graph, forklift1.current_node, target1, forklift1.constraints, forklift1.t_free, speed)
                            if path1:
                                forklift1.schedule = path1
                            else:
                                print(f"Warning: Forklift {forklift1.id} failed to find path to {target1}")
                                forklift1.constraints.pop()
                                task.t_call = t + 50
                                break
                            forklift2.add_constraint({'type': 'vertex', 'node': conflict['node'], 'time': conflict['time']})
                            target2 = forklift2.current_task.start_node if forklift2.current_task else goal
                            path2 = a_star_with_constraints(graph, forklift2.current_node, target2, forklift2.constraints, forklift2.t_free, speed)
                            if path2:
                                forklift2.schedule = path2
                            else:
                                print(f"Warning: Forklift {forklift2.id} failed to find path to {target2}")
                                forklift2.constraints.pop()
                                task.t_call = t + 50
                                break
                        elif conflict['type'] == 'edge':
                            forklift1.add_constraint({'type': 'edge', 'from': conflict['from'], 'to': conflict['to'], 'time': conflict['time']})
                            target1 = goal if forklift1.id == forklift.id else (forklift1.current_task.start_node if forklift1.current_task else goal)
                            path1 = a_star_with_constraints(graph, forklift1.current_node, target1, forklift1.constraints, forklift1.t_free, speed)
                            if path1:
                                forklift1.schedule = path1
                            else:
                                print(f"Warning: Forklift {forklift1.id} failed to find path to {target1}")
                                forklift1.constraints.pop()
                                task.t_call = t + 50
                                break
                            forklift2.add_constraint({'type': 'edge', 'from': conflict['to'], 'to': conflict['from'], 'time': conflict['time']})
                            target2 = forklift2.current_task.start_node if forklift2.current_task else goal
                            path2 = a_star_with_constraints(graph, forklift2.current_node, target2, forklift2.constraints, forklift2.t_free, speed)
                            if path2:
                                forklift2.schedule = path2
                            else:
                                print(f"Warning: Forklift {forklift2.id} failed to find path to {target2}")
                                forklift2.constraints.pop()
                                task.t_call = t + 50
                                break
                    if not forklift.schedule:
                        print(f"Warning: Forklift {forklift.id} schedule is empty after CBS in phase 2 for task {task.task_id}")
                        task.phase = 1
                        task.t_call = t + 50
                        forklift.current_task = None
                        continue
                    forklift.path = [(node, time) for node, time in path[1:]] if len(path) > 1 else []
                    if task not in active_tasks:
                        active_tasks.append(task)
                    print(f"Task {task.task_id} moved to phase 2, heading to goal node {goal}")
                elif task.phase == 2:
                    forklift.t_free = t + 12
                    if task in active_tasks:
                        active_tasks.remove(task)
                    completed_tasks_count += 1
                    forklift.tasks_completed_per_dock[task.dock] += 1
                    forklift.current_task = None
                    forklift.constraints = []
                    nearest = min([node_to_id[coord] for coord in pickup_points.values()], key=lambda n: heuristic(forklift.current_node, n))
                    path = a_star_with_constraints(graph, forklift.current_node, nearest, forklift.constraints, forklift.t_free, speed)
                    if not path:
                        print(f"Warning: No path to nearest pickup for forklift {forklift.id}")
                        forklift.path = []
                        continue
                    forklift.schedule = path
                    for attempt in range(10):
                        conflicts = detect_conflicts(forklifts)
                        if not conflicts:
                            conflict_counts.clear()
                            break
                        conflict = conflicts[0]
                        forklift1_id, forklift2_id = conflict['forklifts']
                        if conflict['type'] == 'vertex':
                            conflict_key = (min(forklift1_id, forklift2_id), max(forklift1_id, forklift2_id), conflict['node'], conflict['time'])
                        elif conflict['type'] == 'edge':
                            conflict_key = (min(forklift1_id, forklift2_id), max(forklift1_id, forklift2_id), (conflict['from'], conflict['to']), conflict['time'])
                        else:
                            print(f"Warning: Unknown conflict type {conflict['type']} after task completion for forklifts {forklift1_id} and {forklift2_id}")
                            forklift.schedule = []
                            break

                        conflict_counts[conflict_key] += 1
                        print(f"Conflict detected after task completion: Forklift {forklift1_id} vs Forklift {forklift2_id}, type={conflict['type']}, time={conflict['time']:.2f}, conflict_count={conflict_counts[conflict_key]}")
                        if conflict_counts[conflict_key] > max_conflict_threshold:
                            print(f"Persistent conflict detected after task completion for forklifts {forklift1_id} and {forklift2_id}. Moving to next task.")
                            forklift.schedule = []
                            for f in forklifts:
                                f.constraints = []
                            conflict_counts.clear()
                            break
                        forklift1 = next(f for f in forklifts if f.id == forklift1_id)
                        forklift2 = next(f for f in forklifts if f.id == forklift2_id)
                        if conflict['type'] == 'vertex':
                            forklift1.add_constraint({'type': 'vertex', 'node': conflict['node'], 'time': conflict['time']})
                            target1 = nearest if forklift1.id == forklift.id or not forklift1.current_task else forklift1.current_task.start_node
                            path1 = a_star_with_constraints(graph, forklift1.current_node, target1, forklift1.constraints, forklift1.t_free, speed)
                            if path1:
                                forklift1.schedule = path1
                            else:
                                print(f"Warning: Forklift {forklift1.id} failed to find path to {target1}")
                                forklift1.constraints.pop()
                                continue
                            forklift2.add_constraint({'type': 'vertex', 'node': conflict['node'], 'time': conflict['time']})
                            target2 = nearest if not forklift2.current_task else forklift2.current_task.start_node
                            path2 = a_star_with_constraints(graph, forklift2.current_node, target2, forklift2.constraints, forklift2.t_free, speed)
                            if path2:
                                forklift2.schedule = path2
                            else:
                                print(f"Warning: Forklift {forklift2.id} failed to find path to {target2}")
                                forklift2.constraints.pop()
                                continue
                        elif conflict['type'] == 'edge':
                            forklift1.add_constraint({'type': 'edge', 'from': conflict['from'], 'to': conflict['to'], 'time': conflict['time']})
                            target1 = nearest if forklift1.id == forklift.id or not forklift1.current_task else forklift1.current_task.start_node
                            path1 = a_star_with_constraints(graph, forklift1.current_node, target1, forklift1.constraints, forklift1.t_free, speed)
                            if path1:
                                forklift1.schedule = path1
                            else:
                                print(f"Warning: Forklift {forklift1.id} failed to find path to {target1}")
                                forklift1.constraints.pop()
                                continue
                            forklift2.add_constraint({'type': 'edge', 'from': conflict['to'], 'to': conflict['from'], 'time': conflict['time']})
                            target2 = nearest if not forklift2.current_task else forklift2.current_task.start_node
                            path2 = a_star_with_constraints(graph, forklift2.current_node, target2, forklift2.constraints, forklift2.t_free, speed)
                            if path2:
                                forklift2.schedule = path2
                            else:
                                print(f"Warning: Forklift {forklift2.id} failed to find path to {target2}")
                                forklift2.constraints.pop()
                                continue
                    if not forklift.schedule:
                        print(f"Warning: Forklift {forklift.id} schedule is empty after CBS for nearest pickup")
                        forklift.path = []
                        continue
                    total_distance = sum(dist for node, dist in graph[forklift.current_node] if node in [n for n, _ in path])
                    time_to_move = total_distance / speed if total_distance else 0
                    forklift.t_free = t + time_to_move
                    forklift.path = [(node, time) for node, time in path[1:]] if len(path) > 1 else []
                    nearest_y = node_coords[nearest][1]
                    nearest_dock = next(dock for dock, vertex in dock_to_vertex.items() if node_coords[vertex][1] == nearest_y)
                    nearest_zone = 1 if nearest_dock >= 10 else 2
                    print(f"Task {task.task_id} completed, Forklift {forklift.id} moving to pickup point {nearest} (Zone {nearest_zone}, Dock {nearest_dock})")

        t += 1

        if len(active_tasks) > max_active_tasks:
            print(f"Error: Active tasks ({len(active_tasks)}) exceeds limit ({max_active_tasks}), terminating simulation")
            break

    if t >= max_simulation_time:
        print("Simulation terminated due to exceeding maximum time limit.")
    else:
        print(f"Simulation completed at time {t} with {completed_tasks_count} tasks finished.")

    dock_avg_wait_times = {}
    for dock in docks:
        total_wait = dock_wait_times[dock]
        num_tasks = dock_task_counts[dock]
        avg_wait = total_wait / num_tasks if num_tasks > 0 else 0
        dock_avg_wait_times[dock] = avg_wait

    return dock_avg_wait_times, task_details, dock_task_counts, initial_task_counts

# Hàm lưu kết quả cho một container
def save_container_results(container_num, dock_avg_wait_times, task_details, dock_task_counts, initial_task_counts, zone_1_docks, zone_2_docks, total_tasks_zone_1, total_tasks_zone_2):
    filename = f"simulation_results_container_{container_num}.txt"
    try:
        with open(filename, "w") as f:
            f.write(f"Simulation Results for Container {container_num}\n")
            f.write("=" * 50 + "\n")
            
            f.write("\nInitial Number of Tasks per Dock (Before Simulation):\n")
            f.write("\nZone 1 (Dock 10-18):\n")
            f.write("Dock | Initial Number of Tasks\n")
            f.write("-" * 30 + "\n")
            for dock in sorted(zone_1_docks, reverse=True):
                f.write(f"{dock:4} | {initial_task_counts[dock]:20}\n")

            f.write("\nZone 2 (Dock 1-9):\n")
            f.write("Dock | Initial Number of Tasks\n")
            f.write("-" * 30 + "\n")
            for dock in sorted(zone_2_docks):
                f.write(f"{dock:4} | {initial_task_counts[dock]:20}\n")

            f.write("\nNumber of Tasks per Dock (After Simulation):\n")
            f.write("\nZone 1 (Dock 10-18):\n")
            f.write("Dock | Number of Tasks\n")
            f.write("-" * 25 + "\n")
            for dock in sorted(zone_1_docks, reverse=True):
                f.write(f"{dock:4} | {dock_task_counts[dock]:15}\n")

            f.write("\nZone 2 (Dock 1-9):\n")
            f.write("Dock | Number of Tasks\n")
            f.write("-" * 25 + "\n")
            for dock in sorted(zone_2_docks):
                f.write(f"{dock:4} | {dock_task_counts[dock]:15}\n")

            f.write(f"\nTotal Tasks in Zone 1: {total_tasks_zone_1}\n")
            f.write(f"Total Tasks in Zone 2: {total_tasks_zone_2}\n")

            f.write("\nAverage Wait Times per Dock:\n")
            f.write("\nZone 1 (Dock 10-18):\n")
            f.write("Dock | Avg Wait Time (s)\n")
            f.write("-" * 25 + "\n")
            for dock in sorted(zone_1_docks, reverse=True):
                f.write(f"{dock:4} | {dock_avg_wait_times[dock]:16.2f}\n")

            f.write("\nZone 2 (Dock 1-9):\n")
            f.write("Dock | Avg Wait Time (s)\n")
            f.write("-" * 25 + "\n")
            for dock in sorted(zone_2_docks):
                f.write(f"{dock:4} | {dock_avg_wait_times[dock]:16.2f}\n")

            f.write("\nTask Processing Details:\n")
            f.write("\nZone 1 (Dock 10-18):\n")
            f.write("Dock | Task ID | Completion Time (s) | Wait Time (s) | Dock Completion Time (s) | Forklift Start Time (s)\n")
            f.write("-" * 100 + "\n")
            for dock in sorted(zone_1_docks, reverse=True):
                for task in task_details[dock]:
                    f.write(f"{dock:4} | {task['task_id']:7} | {task['completion_time']:18.2f} | {task['wait_time']:12.2f} | {task['dock_completion_time']:20.2f} | {task['forklift_start_time']:30.2f}\n")

            f.write("\nZone 2 (Dock 1-9):\n")
            f.write("Dock | Task ID | Completion Time (s) | Wait Time (s) | Dock Completion Time (s) | Forklift Start Time (s)\n")
            f.write("-" * 100 + "\n")
            for dock in sorted(zone_2_docks):
                for task in task_details[dock]:
                    f.write(f"{dock:4} | {task['task_id']:7} | {task['completion_time']:18.2f} | {task['wait_time']:12.2f} | {task['dock_completion_time']:20.2f} | {task['forklift_start_time']:30.2f}\n")
        print(f"Results for Container {container_num} saved to {filename}")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")

# Hàm lưu tóm tắt thời gian chờ trung bình
def save_summary(all_dock_avg_wait_times, zone_1_docks, zone_2_docks):
    filename = "simulation_summary.txt"
    try:
        with open(filename, "w") as f:
            f.write("Summary of Average Wait Times Across 6 Containers\n")
            f.write("=" * 50 + "\n")
            
            f.write("\nZone 1 (Dock 10-18):\n")
            f.write("Dock | Avg Wait Time Across Containers (s)\n")
            f.write("-" * 40 + "\n")
            for dock in sorted(zone_1_docks, reverse=True):
                wait_times = [results[dock] for results in all_dock_avg_wait_times]
                avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
                f.write(f"{dock:4} | {avg_wait:30.2f}\n")

            f.write("\nZone 2 (Dock 1-9):\n")
            f.write("Dock | Avg Wait Time Across Containers (s)\n")
            f.write("-" * 40 + "\n")
            for dock in sorted(zone_2_docks):
                wait_times = [results[dock] for results in all_dock_avg_wait_times]
                avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
                f.write(f"{dock:4} | {avg_wait:30.2f}\n")
        print(f"Summary saved to {filename}")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")

# Hàm lưu bảng thời gian chờ trung bình
def save_wait_time_table(all_dock_avg_wait_times, zone_1_docks, zone_2_docks):
    filename = "wait_time_table.txt"
    try:
        with open(filename, "w") as f:
            f.write("Average Wait Time per Dock Across Containers (seconds)\n")
            f.write("=" * 60 + "\n")
            
            headers = ["Dock", "Container 1", "Container 2", "Container 3", "Container 4", "Container 5", "Container 6", "Average"]
            col_widths = [6, 12, 12, 12, 12, 12, 12, 10]
            
            header_line = "|".join(f"{header:^{w}}" for header, w in zip(headers, col_widths))
            f.write(f"|{header_line}|\n")
            f.write(f"|{'-' * col_widths[0]}|{'-' * col_widths[1]}|{'-' * col_widths[2]}|{'-' * col_widths[3]}|{'-' * col_widths[4]}|{'-' * col_widths[5]}|{'-' * col_widths[6]}|{'-' * col_widths[7]}|\n")
            
            f.write("\nZone 1 (Dock 10-18):\n")
            for dock in sorted(zone_1_docks, reverse=True):
                wait_times = [results.get(dock, 0) for results in all_dock_avg_wait_times]
                avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
                row = [f"Dock {dock}", *[f"{wt:>6.2f}" for wt in wait_times], f"{avg_wait:>6.2f}"]
                row_line = "|".join(f"{cell:^{w}}" for cell, w in zip(row, col_widths))
                f.write(f"|{row_line}|\n")
            
            f.write(f"|{'-' * col_widths[0]}|{'-' * col_widths[1]}|{'-' * col_widths[2]}|{'-' * col_widths[3]}|{'-' * col_widths[4]}|{'-' * col_widths[5]}|{'-' * col_widths[6]}|{'-' * col_widths[7]}|\n")
            
            f.write("\nZone 2 (Dock 1-9):\n")
            for dock in sorted(zone_2_docks):
                wait_times = [results.get(dock, 0) for results in all_dock_avg_wait_times]
                avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
                row = [f"Dock {dock}", *[f"{wt:>6.2f}" for wt in wait_times], f"{avg_wait:>6.2f}"]
                row_line = "|".join(f"{cell:^{w}}" for cell, w in zip(row, col_widths))
                f.write(f"|{row_line}|\n")
            
            f.write(f"|{'-' * col_widths[0]}|{'-' * col_widths[1]}|{'-' * col_widths[2]}|{'-' * col_widths[3]}|{'-' * col_widths[4]}|{'-' * col_widths[5]}|{'-' * col_widths[6]}|{'-' * col_widths[7]}|\n")

        print("\nAverage Wait Time per Dock Across Containers (seconds):")
        print("=" * 60)
        print(f"|{header_line}|")
        print(f"|{'-' * col_widths[0]}|{'-' * col_widths[1]}|{'-' * col_widths[2]}|{'-' * col_widths[3]}|{'-' * col_widths[4]}|{'-' * col_widths[5]}|{'-' * col_widths[6]}|{'-' * col_widths[7]}|")
        print("\nZone 1 (Dock 10-18):")
        for dock in sorted(zone_1_docks, reverse=True):
            wait_times = [results.get(dock, 0) for results in all_dock_avg_wait_times]
            avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
            row = [f"Dock {dock}", *[f"{wt:>6.2f}" for wt in wait_times], f"{avg_wait:>6.2f}"]
            row_line = "|".join(f"{cell:^{w}}" for cell, w in zip(row, col_widths))
            print(f"|{row_line}|")
        print(f"|{'-' * col_widths[0]}|{'-' * col_widths[1]}|{'-' * col_widths[2]}|{'-' * col_widths[3]}|{'-' * col_widths[4]}|{'-' * col_widths[5]}|{'-' * col_widths[6]}|{'-' * col_widths[7]}|")
        print("\nZone 2 (Dock 1-9):")
        for dock in sorted(zone_2_docks):
            wait_times = [results.get(dock, 0) for results in all_dock_avg_wait_times]
            avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
            row = [f"Dock {dock}", *[f"{wt:>6.2f}" for wt in wait_times], f"{avg_wait:>6.2f}"]
            row_line = "|".join(f"{cell:^{w}}" for cell, w in zip(row, col_widths))
            print(f"|{row_line}|")
        print(f"|{'-' * col_widths[0]}|{'-' * col_widths[1]}|{'-' * col_widths[2]}|{'-' * col_widths[3]}|{'-' * col_widths[4]}|{'-' * col_widths[5]}|{'-' * col_widths[6]}|{'-' * col_widths[7]}|")
        
        print(f"\nTable saved to {filename}")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")

# Chạy mô phỏng cho 6 container
num_containers = 6
all_dock_avg_wait_times = []
zone_1_docks = [dock for dock in range(1, 19) if dock >= 10]
zone_2_docks = [dock for dock in range(1, 19) if dock < 10]

for container in range(1, num_containers + 1):
    print(f"\nSimulating Container {container}...")
    dock_avg_wait_times, task_details, dock_task_counts, initial_task_counts = simulate()
    total_tasks_zone_1 = sum(dock_task_counts[dock] for dock in zone_1_docks)
    total_tasks_zone_2 = sum(dock_task_counts[dock] for dock in zone_2_docks)
    print(f"\nResults for Container {container}:")
    print("\nNumber of Tasks per Dock (After Simulation):")
    print("Zone 1 (Dock 10-18):")
    print("Dock | Number of Tasks")
    print("-" * 25)
    for dock in sorted(zone_1_docks, reverse=True):
        print(f"{dock:4} | {dock_task_counts[dock]:15}")
    print("\nZone 2 (Dock 1-9):")
    print("Dock | Number of Tasks")
    print("-" * 25)
    for dock in sorted(zone_2_docks):
        print(f"{dock:4} | {dock_task_counts[dock]:15}")
    print(f"\nTotal Tasks in Zone 1: {total_tasks_zone_1}")
    print(f"Total Tasks in Zone 2: {total_tasks_zone_2}")
    print("\nAverage Wait Times per Dock:")
    print("Zone 1 (Dock 10-18):")
    print("Dock | Avg Wait Time (s)")
    print("-" * 25)
    for dock in sorted(zone_1_docks, reverse=True):
        print(f"{dock:4} | {dock_avg_wait_times[dock]:16.2f}")
    print("\nZone 2 (Dock 1-9):")
    print("Dock | Avg Wait Time (s)")
    print("-" * 25)
    for dock in sorted(zone_2_docks):
        print(f"{dock:4} | {dock_avg_wait_times[dock]:16.2f}")
    print("\nTask Processing Details:")
    print("\nZone 1 (Dock 10-18):")
    print("Dock | Task ID | Completion Time (s) | Wait Time (s) | Dock Completion Time (s) | Forklift Start Time (s)")
    print("-" * 100)
    for dock in sorted(zone_1_docks, reverse=True):
        for task in task_details[dock]:
            print(f"{dock:4} | {task['task_id']:7} | {task['completion_time']:18.2f} | {task['wait_time']:12.2f} | {task['dock_completion_time']:20.2f} | {task['forklift_start_time']:22.2f}")
    print("\nZone 2 (Dock 1-9):")
    print("Dock | Task ID | Completion Time (s) | Wait Time (s) | Dock Completion Time (s) | Forklift Start Time (s)")
    print("-" * 100)
    for dock in sorted(zone_2_docks):
        for task in task_details[dock]:
            print(f"{dock:4} | {task['task_id']:7} | {task['completion_time']:18.2f} | {task['wait_time']:12.2f} | {task['dock_completion_time']:20.2f} | {task['forklift_start_time']:22.2f}")
    
    all_dock_avg_wait_times.append(dock_avg_wait_times)
    save_container_results(container, dock_avg_wait_times, task_details, dock_task_counts, initial_task_counts, zone_1_docks, zone_2_docks, total_tasks_zone_1, total_tasks_zone_2)

save_summary(all_dock_avg_wait_times, zone_1_docks, zone_2_docks)
save_wait_time_table(all_dock_avg_wait_times, zone_1_docks, zone_2_docks)