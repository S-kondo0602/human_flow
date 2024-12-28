import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import numpy as np

class Pedestrian(Agent):
    def __init__(self, unique_id, model, movement_type):
        super().__init__(unique_id, model)
        self.movement_type = movement_type  # "cross_up", "cross_down", or "photograph"
        self.has_crossed = False
        if self.movement_type == "photograph":
            self.photographing_time = self.random.randint(10, 15)  # 写真撮影にかかるランダムな遅延時間

    def step(self):
        if self.movement_type == "cross_up":
            self.move_up()
        elif self.movement_type == "cross_down":
            self.move_down()
        elif self.movement_type == "photograph":
            self.photograph_and_move()

    def move_up(self):
        if self.pos[1] == 0:
            self.model.grid.move_agent(self, (self.pos[0], self.model.grid.height - 1))  # 下に戻す
        else:
            next_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            possible_moves_up = [pos for pos in next_moves if pos[1] < self.pos[1]]  # 上方向への移動
            straight_up = [pos for pos in possible_moves_up if pos[0] == self.pos[0]]  # 真上への移動
            diagonal_moves = [pos for pos in possible_moves_up if pos[0] != self.pos[0]]  # 斜め方向への移動

            if straight_up and self.model.grid.is_cell_empty(straight_up[0]):
                self.model.grid.move_agent(self, straight_up[0])
            elif diagonal_moves:
                diagonal_moves = [move for move in diagonal_moves if self.model.grid.is_cell_empty(move)]
                if diagonal_moves:
                    next_move = self.random.choice(diagonal_moves)
                    self.model.grid.move_agent(self, next_move)

    def move_down(self):
        if self.pos[1] == self.model.grid.height - 1:
            self.model.grid.move_agent(self, (self.pos[0], 0))  # 上に戻す
        else:
            next_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            possible_moves_down = [pos for pos in next_moves if pos[1] > self.pos[1]]  # 下方向への移動
            straight_down = [pos for pos in possible_moves_down if pos[0] == self.pos[0]]  # 真下への移動
            diagonal_moves = [pos for pos in possible_moves_down if pos[0] != self.pos[0]]  # 斜め方向への移動

            if straight_down and self.model.grid.is_cell_empty(straight_down[0]):
                self.model.grid.move_agent(self, straight_down[0])
            elif diagonal_moves:
                diagonal_moves = [move for move in diagonal_moves if self.model.grid.is_cell_empty(move)]
                if diagonal_moves:
                    next_move = self.random.choice(diagonal_moves)
                    self.model.grid.move_agent(self, next_move)

    def photograph_and_move(self):
        # 右2列でのみ移動
        if self.pos[0] >= self.model.grid.width - 10:
            if self.photographing_time > 0:
                self.photographing_time -= 1  # 写真撮影中
            else:
                self.has_crossed = True
                # 撮影終了後は上下方向に移動
                next_moves = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
                possible_moves = [pos for pos in next_moves if pos[0] >= self.model.grid.width - 10]  # 右2列内の移動
                possible_moves = [move for move in possible_moves if self.model.grid.is_cell_empty(move)]
                if possible_moves:
                    next_move = self.random.choice(possible_moves)
                    self.model.grid.move_agent(self, next_move)

class CrosswalkModel(Model):
    def __init__(self, width, height, N):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, False)  # トーラスではない境界
        self.schedule = RandomActivation(self)

        # エージェントを作成
        for i in range(self.num_agents):
            movement_type = self.random.choice(["cross_up", "cross_down", "photograph"])
            agent = Pedestrian(i, self, movement_type)
            self.schedule.add(agent)
            if movement_type == "photograph":
                # 写真撮影者は右2列に配置
                x = self.random.randint(self.grid.width - 10, self.grid.width - 1)
            else:
                # それ以外のエージェントはランダムな位置に配置
                x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def step(self):
        self.schedule.step()

# Visualization
def plot_agents(model):
    grid_data = np.zeros((model.grid.height, model.grid.width), dtype=int)
    
    for (content, pos) in model.grid.coord_iter():
        x, y = pos
        for obj in content:
            if isinstance(obj, Pedestrian):
                if obj.movement_type == "cross_up":
                    grid_data[y][x] = 1  # 青で上方向に移動するエージェント
                elif obj.movement_type == "cross_down":
                    grid_data[y][x] = 3  # 赤で下方向に移動するエージェント
                elif obj.movement_type == "photograph":
                    grid_data[y][x] = 2  # 緑で写真を撮るエージェント

    return grid_data

# Update function for animation
def update(frame_number, model, im):
    grid_data = plot_agents(model)
    
    im.set_data(grid_data)
    model.step()
    return [im]

# モデルを初期化
model = CrosswalkModel(10, 22, 54)

# プロット設定
fig, ax = plt.subplots()
grid_data = plot_agents(model)

# カスタムカラーマップを使用
im = ax.imshow(grid_data, cmap=plt.cm.get_cmap('coolwarm', 4), origin='upper', interpolation='nearest')

# アニメーション設定
ani = animation.FuncAnimation(fig, update, fargs=(model, im), frames=120, interval=100, blit=False)

# アニメーションを表示
plt.show()
