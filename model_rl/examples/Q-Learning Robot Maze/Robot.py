import random

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        
        if self.testing:
            # TODO 1. No random choice when testing
            self.epsilon = 0
        else:
            # TODO 2. Update parameters when learning
            #self.epsilon = self.epsilon * 0.99
            tolerance = 0.001
            import math
            if self.epsilon > tolerance:
                self.epsilon = math.cos(math.pi/2/300*self.t) #cos 凸衰减函数，前期衰减慢多探索，后期衰减快少探索，#认为每次运行300步左右时应该要衰减到0
                self.t +=1
            
            else:
                self.epsilon = tolerance

            
        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        #state = self.state
        if state in self.Qtable:
            pass
        else:
            self.Qtable[state] = {'u':0, 'd':0, 'l':0, 'r':0}
            #self.Qtable.setdefault(state, {a: 0.0 for a in self.valid_actions})

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            #if random.random() < self.epsilon: # 以某一概率
                #return True
            #else:
                #return False
            return random.random() < self.epsilon


        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                action = random.choice(self.valid_actions)
                return action
            else:
                # TODO 7. Return action with highest q value
                return sorted(self.Qtable[self.state].items(), key=lambda e:e[1], reverse=True)[0][0]
                #return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        elif self.testing:
            # TODO 7. choose action with highest q value
            action = sorted(self.Qtable[self.state].items(), key=lambda e:e[1], reverse=True)[0][0]
            #action =max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        else:
            # TODO 6. Return random choose aciton
            return random.choice(self.valid_actions)

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            maxQ_next = sorted(self.Qtable[next_state].items(), key=lambda e:e[1], reverse=True)[0][1]
            #maxQ_next = max(self.Qtable[self.state].values())
            self.Qtable[self.state][action] = (1-self.alpha)*self.Qtable[self.state][action] + self.alpha*(r+self.gamma*maxQ_next)
            # TODO 8. When learning, update the q table according to the given rules

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # 得到当前状态

        self.create_Qtable_line(self.state) # 给当前状态创建Qtable(初始为0)

        action = self.choose_action() # 为这个状态选择行为
        reward = self.maze.move_robot(action) # 根据行为移动机器人

        next_state = self.sense_state() # 得到下一个状态
        self.create_Qtable_line(next_state) # 为下一个状态创建Qtable

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # 根据奖励、行为和下个状态的Qtable，更新当前状态的Qtable
            self.update_parameter() # 更新参数

        return action, reward
