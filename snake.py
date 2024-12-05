# importing libraries
import pygame
import random
import torch
import math
import copy
from sklearn import preprocessing
from ai import superAi
from ai import module_weight_average
from ai import aiTrainer

snake_speed = 15

# Window size
window_x = 200
window_y = 150
# window_x = 720
# window_y = 480


def is_collision(snake_x, snake_y, snake_body):

    # Game Over conditions
    if snake_x < 0 or snake_x > window_x-10: 
        return True
    if snake_y < 0 or snake_y > window_y-10:
        return True
    for block in snake_body[1:]:
        if snake_position[0] == block[0] and snake_position[1] == block[1]:
            return True
    return False

def calc_inputs(fruit_x:int, fruit_y:int, snake_x:int, snake_y:int, snake_body:list, snake_direct:str):

   
    # vision for seeing snake body
    point_l = (snake_x - 10, snake_y)
    point_r = (snake_x + 10, snake_y)
    point_u = (snake_x, snake_y - 10)
    point_d = (snake_x, snake_y + 10)
        
    dir_l =  snake_direct == "LEFT"
    dir_r =  snake_direct == "RIGHT"
    dir_u =  snake_direct == "UP"
    dir_d =  snake_direct == "DOWN" 

    inputs = [
        # danger straight
        (dir_r and is_collision(point_r[0], point_r[1], snake_body)) or 
        (dir_l and is_collision(point_l[0], point_l[1], snake_body)) or 
        (dir_u and is_collision(point_u[0], point_u[1], snake_body)) or 
        (dir_d and is_collision(point_d[0], point_d[1], snake_body)),

        # danger right
        (dir_u and is_collision(point_r[0], point_r[1], snake_body)) or 
        (dir_d and is_collision(point_l[0], point_l[1], snake_body)) or 
        (dir_l and is_collision(point_u[0], point_u[1], snake_body)) or 
        (dir_r and is_collision(point_d[0], point_d[1], snake_body)),

        # danger left
        (dir_d and is_collision(point_r[0], point_r[1], snake_body)) or 
        (dir_u and is_collision(point_l[0], point_l[1], snake_body)) or 
        (dir_r and is_collision(point_u[0], point_u[1], snake_body)) or 
        (dir_l and is_collision(point_d[0], point_d[1], snake_body)),
        
        # move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        
        # food location 
        fruit_x < snake_x,  # food left
        fruit_x > snake_x,  # food right
        fruit_y < snake_y,  # food up
        fruit_y > snake_y  # food down
        ]

    return inputs

# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# Initialising pygame
pygame.init()

# Initialise game window
pygame.display.set_caption('GeeksforGeeks Snakes')
game_window = pygame.display.set_mode((window_x, window_y))

# FPS (frames per second) controller
fps = pygame.time.Clock()

# defining snake default position
snake_position = [100, 50]

# defining first 4 blocks of snake body
snake_body = [[100, 50],
              [90, 50],
              [80, 50],
              [70, 50],
              [60, 50]]
# fruit position
fruit_position = [random.randrange(1, (window_x//10)) * 10, 
                  random.randrange(1, (window_y//10)) * 10]

fruit_spawn = True

# setting default snake direction towards
# right
direction = 'RIGHT'
change_to = direction

# initial score
score = 0
temp_score = 0
reward = 0
numepochs = 1
temp_epchos = 0
move_n = 0
max_move_n = ((window_x*window_y)/100) * (1-(1/len(snake_body)))

# snake ai
input_len = len(calc_inputs(fruit_position[0], fruit_position[1],
                          snake_position[0], snake_position[1], snake_body, change_to))
snake_ai = superAi(input_len, dropout_rate=0.2)
ai_move_dict = {0:"LEFT", 1:"UP", 2:"RIGHT", 3:"DOWN"}
train_data_dict = {"inputs":[], "actions":[], "rewards":[], "pred_inputs":[], "game_status":[]}
good_snakes = []

# displaying Score function
def show_score(color, font, size):
  
    # creating font object score_font
    score_font = pygame.font.SysFont(font, size)
    
    # create the display surface object 
    # score_surface
    score_surface = score_font.render('Score : ' + str(score), True, color)
    
    # create a rectangular object for the text
    # surface object
    score_rect = score_surface.get_rect()
    
    # displaying text
    game_window.blit(score_surface, score_rect)

def show_other(color, font, size):
    # creating font object epochs_font
    epochs_font = pygame.font.SysFont(font, size)
    reward_n_font = pygame.font.SysFont(font, size)

    # create the display surface object 
    # epochs_surface
    epochs_surface = epochs_font.render('epochs : ' + str(numepochs), True, color)
    reward_n_surface = reward_n_font.render('reward : ' + str(round(reward,3)), True, color)   
    move_n_surface = reward_n_font.render('move : ' + str(move_n), True, color)   

    # create a rectangular object for the text
    # surface object
    epochs_rect = epochs_surface.get_rect()
    reward_n_rect = reward_n_surface.get_rect()

    new_pos = list(reward_n_rect.bottomleft)
    new_pos = tuple([new_pos[0], new_pos[1]+20])
    new_pos2 = tuple([new_pos[0], new_pos[1]+20])

    # displaying text
    game_window.blit(epochs_surface, epochs_rect.bottomleft)
    game_window.blit(reward_n_surface, new_pos)
    game_window.blit(move_n_surface, new_pos2)

# game over function
def game_over(numepochs:int, train_data_dict:dict, snake_body):
    numepochs += 1

    try:
        # give punish when game over
        train_data_dict["rewards"] = train_data_dict["rewards"][0:-1] + [-math.pow(max_move_n, (max_move_n-len(snake_body))/max_move_n)/100]
        # game over status
        train_data_dict["game_status"] = train_data_dict["game_status"][0:-1] + ["GameOver"]

        trainer = aiTrainer(snake_ai, .001, .1) 
        trainer.train_step( train_data_dict["inputs"], train_data_dict["actions"],
                         train_data_dict["rewards"],train_data_dict["game_status"])
    except IndexError:
        pass
    except RuntimeError:
        if len(train_data_dict["inputs"]) ==0:
            pass
        else:
            print(train_data_dict["inputs"])

    #reset the game
    move_n = 0
    snake_position = [100, 50]
    snake_body = [[100, 50],
                [90, 50],
                [80, 50],
                [70, 50],
                [60, 50]]
    fruit_position = [random.randrange(1, (window_x//10)) * 10, 
                    random.randrange(1, (window_y//10)) * 10]
    fruit_spawn = True
    direction = 'RIGHT'
    change_to = direction
    # train_data_dict = {"inputs":[], "actions":[], "rewards":[], "pred_inputs":[], "game_status":[]}
    return train_data_dict,move_n, 0, numepochs, snake_position, snake_body, fruit_position, fruit_spawn,change_to, numepochs

def closer_reward(old_position, new_position, fruit_position):
    old_len = math.sqrt((old_position[0]-fruit_position[0])**2 + (old_position[1]-fruit_position[1])**2)
    new_len = math.sqrt((new_position[0]-fruit_position[0])**2 + (new_position[1]-fruit_position[1])**2)
    return -1/len(snake_body) if old_len < new_len else 1/len(snake_body)

def pasue():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                return 
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                quit()
    

# Main Function
while True:
    
    # handling key events
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                change_to = 'UP'
            if event.key == pygame.K_DOWN:
                change_to = 'DOWN'
            if event.key == pygame.K_LEFT:
                change_to = 'LEFT'
            if event.key == pygame.K_RIGHT:
                change_to = 'RIGHT'
            if event.key == pygame.K_ESCAPE:
                quit()
            if event.key == pygame.K_p:
                pasue()
            if event.key == pygame.K_r:
                snake_ai = superAi(input_len, dropout_rate=0.2)
                print("new snake")
 
    input_s = calc_inputs(fruit_position[0], fruit_position[1],
                          snake_position[0], snake_position[1], snake_body, direction)
    # ai control
    action = snake_ai(input_s)
    change_to = ai_move_dict[torch.argmax(action).item()]
    old_position = snake_position.copy()


    # If two keys pressed simultaneously
    # we don't want snake to move into two 
    # directions simultaneously
    if change_to == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP':
        direction = 'DOWN'
    if change_to == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if change_to == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'

    # Moving the snake
    if direction == 'UP':
        snake_position[1] -= 10
    if direction == 'DOWN':
        snake_position[1] += 10
    if direction == 'LEFT':
        snake_position[0] -= 10
    if direction == 'RIGHT':
        snake_position[0] += 10

    
    # Snake body growing mechanism
    # if fruits and snakes collide then scores
    snake_body.insert(0, list(snake_position))
    if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
        score = 10
        fruit_spawn = False
        reward = (len(snake_body) / (max_move_n))
        max_move_n = ((window_x*window_y)/100) * (1-(1/len(snake_body)))
        move_n = 0
    else:
        reward = closer_reward(old_position, snake_position, fruit_position)
        snake_body.pop()
        move_n += 1
        
    if not fruit_spawn:
        fruit_position = [random.randrange(1, (window_x//10)) * 10, 
                          random.randrange(1, (window_y//10)) * 10]

    # highest score
    if score > temp_score:
        temp_score = score
        temp_epchos = numepochs
        good_snakes.append(copy.deepcopy(snake_ai))
        print(f"High score: {score}")
 

    fruit_spawn = True
    game_window.fill(black)
    
    for i, (pos_x, pos_y) in enumerate(snake_body):
        if i == 0:
            pygame.draw.circle(game_window, green,
                         (pos_x+5,pos_y+5),5,3)
        else:
            pygame.draw.rect(game_window, green,
                         pygame.Rect(pos_x, pos_y, 10, 10))
    pygame.draw.rect(game_window, red, pygame.Rect(
        fruit_position[0], fruit_position[1], 10, 10))

    # memorize step and others
    train_data_dict["inputs"] += [input_s]
    train_data_dict["actions"] += [action]
    train_data_dict["rewards"] += [reward]
    train_data_dict["game_status"] += ["Go"]

    # trainer = aiTrainer(snake_ai, .001, .1) 
    # trainer.train_step( train_data_dict["inputs"], train_data_dict["actions"],
    #                      train_data_dict["rewards"],train_data_dict["game_status"])   

    # train short
    # trainer = aiTrainer(snake_ai, .05, .05) 
    # trainer.train_step( [input_s], [action],
                        # [reward], ["Go"])
   
    # Game Over conditions
    if snake_position[0] < 0 or snake_position[0] > window_x-10: 
        train_data_dict, move_n, \
        score, numepochs, \
        snake_position, snake_body, \
        fruit_position, fruit_spawn, \
        change_to, numepochs = game_over(numepochs, train_data_dict, snake_body)
        reward = 0

    if snake_position[1] < 0 or snake_position[1] > window_y-10:
        train_data_dict, move_n,\
        score, numepochs, \
        snake_position, snake_body, \
        fruit_position, fruit_spawn, \
        change_to, numepochs = game_over(numepochs, train_data_dict, snake_body)
        reward = 0
    
    if move_n > max_move_n:
        train_data_dict, move_n,\
        score, numepochs, \
        snake_position, snake_body, \
        fruit_position, fruit_spawn, \
        change_to, numepochs = game_over(numepochs, train_data_dict, snake_body)
        reward = 0

    # Touching the snake body
    for block in snake_body[1:]:
        if snake_position[0] == block[0] and snake_position[1] == block[1]:
            train_data_dict, move_n, \
            score, numepochs, \
            snake_position, snake_body, \
            fruit_position, fruit_spawn, \
            change_to, numepochs = game_over(numepochs, train_data_dict, snake_body)
            reward = 0

    # displaying score continuously
    show_score(white, 'times new roman', 20)
    show_other(white, 'times new roman', 20)
    # Refresh game screen
    pygame.display.update()

    # kill the snake if no imporvement over 400 epoches
    if score <= temp_score and (numepochs - temp_epchos)>400:
        snake_ai = superAi(input_len, .02)
        numepochs = 0
        temp_epchos = 0
        temp_score = 0

    # mix with good snakes    
    if len(good_snakes) == 10:
        snake_ai = module_weight_average(good_snakes)

    # Frame Per Second /Refresh Rate
    fps.tick(snake_speed)