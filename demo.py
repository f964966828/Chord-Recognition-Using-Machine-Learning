import os
from utils import get_params, load_data, PitchClassProfiler
from sklearn.neighbors import KNeighborsClassifier

import pygame
import matplotlib
import matplotlib.backends.backend_agg as agg
import pylab
import time
from pygame.locals import *

params = get_params()
train_data_path = params["train_data_path"]
train_json_name = params["train_json_name"]
test_data_path = params["test_data_path"]
test_json_name = params["test_json_name"]
test_file_name = params["test_file_name"]

unit = params["unit"]

train_json_path = os.path.join(train_data_path, train_json_name)
test_json_path = os.path.join(test_data_path, test_json_name)
test_file_path = os.path.join(test_data_path, test_file_name)


def playsound(soundfile):
    """Play sound through default mixer channel in blocking manner.
       This will load the whole sound into memory before playback
    """    
    #pygame.init()
    pygame.mixer.init()
    sound = pygame.mixer.Sound(soundfile)
    clock = pygame.time.Clock()
    sound.play()

def animation():
    global start_time
    global y_pred
    global y_true

    current_time = (time.time() - start_time) * 1.15

    fig = pylab.figure(figsize=[15, 5], dpi=100)
    ax = fig.gca()
    ax.plot(ptc.samples()[:-1 * (len(ptc.samples()) % unit) - unit])
    ax.plot([current_time * unit, current_time * unit], [-22000, 22000])
    ax.set_ylim(-45000, 22000)

    for i, y in enumerate(y_pred):
        if i == current_time // 1:
            ax.text(i * unit, -33000, y, fontsize=18)
        else:
            ax.text(i * unit, -38000, y, fontsize=12)

        ax.text(i * unit, -44000, y_true[i], fontsize=12)

    # set frame invisible
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()

    screen = pygame.display.get_surface()
    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (0,0))


def listen():
    """ quit the window when user presses the red x button of the window """
    [pygame.quit() for event in pygame.event.get() if event.type == pygame.QUIT]

def mainloop():
    """ This function runs until users quit the window """
    while (time.time() - start_time < len(ptc.samples()) // unit - 2):
        listen() # this calls the function to quit the window
        screen.fill(0)
        animation() # This will call the animation
        clock.tick(5)
        pygame.display.flip()

matplotlib.use("Agg")

ptc = PitchClassProfiler(test_file_path)  

if __name__ == "__main__":

    X, y, m = load_data(train_json_path, return_mapping=True)
    X_test, _, _ = load_data(test_json_path, return_mapping=True)

    #Create Classifier
    model = KNeighborsClassifier(n_neighbors=3)
    #Train the model using the training sets
    model.fit(X, y)
    #Predict the response for test dataset
    y_pred = model.predict(X_test)
    y_pred = [m[y] for y in y_pred]
    print(len(y_pred))
    y_true = (["G"]*3 + ["D"]*3 + ["Em"]*3 + ["C"]*3) * 3

    pygame.init()
    start_time = time.time()
    playsound(test_file_path)
    screen = pygame.display.set_mode((1500, 500), DOUBLEBUF)
    clock = pygame.time.Clock()

    mainloop()
