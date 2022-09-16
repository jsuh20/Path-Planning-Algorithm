# path planning algorithm
import os

from flask import Flask, redirect, render_template, request, session, url_for
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import sqrt
from io import BytesIO
import base64
import matplotlib.animation as animation
# input: 4 vertex points of rectangular field
# minimum turning radius
# Width of tractor
# Starting point on plane

# class cell:
#     def __init__(self, row, col, width, total_rows):
#         self.cleaned = false
#         self.row = row
#         self.col = col
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html", message="Boustrophedon Path Planning")

@app.route("/login", methods=['POST', 'GET'])
def login():
    if request.method == "POST":
        session["start_point"] = request.form["start_point"]
        session["vertex"] = request.form["vertex"]
        session["turn_rad"] = request.form["turn_rad"]
        session["direction"] = request.form["direction"]
        print("hi there")
        return redirect(url_for('presentGraph', start=session["start_point"], vertex=session["vertex"], turn_rad=session["turn_rad"], direction=session["direction"]))

@app.route("/presentGraph/<start>/<vertex>/<turn_rad>/<direction>/projection")
def presentGraph(start, vertex, turn_rad, direction):
    print(start," ", vertex," ", turn_rad," ", direction)
    graph = draw_grid(start, vertex, turn_rad, direction)
    print("hi")
    return render_template("projection.html", message="Boustrophedon Path Planning", plot_url=graph)

def update_list(path_list, cleaned_pts, uncleaned_pts, current):
    path_list_len = len(path_list)
    cleaned_len = len(cleaned_pts)
    path_list.loc[path_list_len] = current
    if not uncleaned_pts[(uncleaned_pts.x == current[0]) & (uncleaned_pts.y == current[1])].empty:
        cleaned_pts.loc[cleaned_len] = current  # change status to cleaned
        uncleaned_pts.drop(uncleaned_pts.index[(uncleaned_pts['x'] == current[0]) & (uncleaned_pts['y'] == current[1])],
                           inplace=True)
    return path_list, cleaned_pts, uncleaned_pts

def cov_plan_algo(start, vert_pts, turn_rad, direction):  # tlv = top left vertex
    # returns path dataframe
    # initialize relevant lists (cleaned and uncleaned)
    # use starting point and find direction
    # loop until all points are cleaned
    cleaned_pts = pd.DataFrame({"x": [start[0]], "y": [start[1]]})
    path_list = pd.DataFrame({"x": [start[0]], "y": [start[1]]})
    uncleaned_pts = pd.DataFrame({"x": [], "y": []})
    col = vert_pts["x"].max()
    min_x = vert_pts["x"].min()
    row = vert_pts["y"].max()
    min_y = vert_pts["y"].min()
    for i in range(min_x, col+1): # initialize uncleaned list
        for j in range(min_y, row+1):
            if i is start[0] and j is start[1]:  # appends everything except starting position
                continue
            to_append = [i, j]
            df_len = len(uncleaned_pts)
            uncleaned_pts.loc[df_len] = to_append
    # find direction for after starting point (total 4 conditions)
    current = [start[0], start[1]]
    if direction == "right" or direction == "left":
        if direction == "right":
            while col - current[0] > turn_rad:  # initial straight
                current[0] += 1
                path_list, cleaned_pts, uncleaned_pts = update_list(path_list,cleaned_pts,uncleaned_pts,current)
        elif direction == "left":
            while current[0]-min_x > turn_rad:  # initial straight
                current[0] -= 1
                path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
        while len(uncleaned_pts) != 0:
                # path_list[(path_list.x == 1) & (path_list.y == 0)].empty
            if cleaned_pts[(cleaned_pts.x == current[0]+1) & (cleaned_pts.y == current[1])].empty and col - current[0] > turn_rad: # straight
                while col - current[0] > turn_rad:
                    current[0] += 1   # variable need to automatically know which way
                    path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
            elif cleaned_pts[(cleaned_pts.x == current[0]-1) & (cleaned_pts.y == current[1])].empty and current[0] - min_x >= turn_rad: # straight (changed from col - current[0]
                while current[0] - min_x > turn_rad:
                    current[0] -= 1
                    path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
            if current[1] + turn_rad*2 <= row and cleaned_pts[cleaned_pts.y == current[1] + turn_rad*2].empty: #turn
                mid_pt = [current[0], current[1] + turn_rad]
                ch_pt = [current[0], current[1] + turn_rad*2] # needed
                for y in range(current[1], ch_pt[1]+1): # add points of semicircle
                    if current[0] - min_x <= turn_rad:
                        formula = -1 * (sqrt(turn_rad ** 2 - (y - mid_pt[1]) ** 2)) + mid_pt[0]
                    elif col - current[0] <= turn_rad:
                        formula = sqrt(turn_rad ** 2 - (y - mid_pt[1]) ** 2) + mid_pt[0]
                    current = [formula, y]
                    path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
                current[1] = ch_pt[1]
            elif current[1] - turn_rad*2 >= min_y and cleaned_pts[cleaned_pts.y == current[1] - turn_rad*2].empty:
                mid_pt = [current[0], current[1] - turn_rad]
                ch_pt = [current[0], current[1] - turn_rad * 2]
                for y in range(current[1], ch_pt[1]-1, -1): # add points of semicircle
                    if current[0] - min_x <= turn_rad:
                        formula = -1 * (sqrt(turn_rad ** 2 - (y - mid_pt[1]) ** 2)) + mid_pt[0]
                    elif col - current[0] <= turn_rad:
                        formula = sqrt(turn_rad ** 2 - (y - mid_pt[1]) ** 2) + mid_pt[0]
                    current = [formula, y]
                    path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
                current[1] = ch_pt[1]
            else:
                return path_list
    elif direction == "top" or direction == "bottom":
        if direction == "top":
            while row - current[1] > turn_rad:  # initial straight
                current[1] += 1
                path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
        elif direction == "bottom":
            while current[1] - min_y > turn_rad:  # initial straight
                current[1] -= 1
                path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
        while len(uncleaned_pts) != 0:
            # path_list[(path_list.x == 1) & (path_list.y == 0)].empty
            if cleaned_pts[(cleaned_pts.x == current[0]) & (cleaned_pts.y == current[1] + 1)].empty and row - current[1] > turn_rad:  # straight
                while row - current[1] > turn_rad:
                    current[1] += 1  # variable need to automatically know which way
                    path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
            elif cleaned_pts[(cleaned_pts.x == current[0]) & (cleaned_pts.y == current[1] - 1)].empty and  current[1] - min_x >= turn_rad:  # straight
                while current[1] - min_x > turn_rad:
                    current[1] -= 1
                    path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
            if current[0] + turn_rad * 2 <= col and cleaned_pts[cleaned_pts.x == current[0] + turn_rad * 2].empty:  # turn
                mid_pt = [current[0] + turn_rad, current[1]]
                ch_pt = [current[0] + turn_rad * 2, current[1]]  # needed
                for x in range(current[0], ch_pt[0] + 1):  # add points of semicircle
                    if current[1] - min_y <= turn_rad:
                        formula = -1 * (sqrt(turn_rad ** 2 - (x - mid_pt[0]) ** 2)) + mid_pt[1]
                    elif row - current[1] <= turn_rad:
                        formula = sqrt(turn_rad ** 2 - (x - mid_pt[0]) ** 2) + mid_pt[1]
                    current = [x, formula]
                    path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
                current[0] = ch_pt[0]
            elif current[0] - turn_rad * 2 >= min_x and cleaned_pts[cleaned_pts.x == current[0] - turn_rad * 2].empty:
                mid_pt = [current[0] - turn_rad, current[1]]
                ch_pt = [current[0] - turn_rad * 2, current[1]]
                for x in range(current[0], ch_pt[0] - 1, -1):  # add points of semicircle
                    if current[1] - min_y <= turn_rad:
                        formula = -1 * (sqrt(turn_rad ** 2 - (x - mid_pt[0]) ** 2)) + mid_pt[1]
                    elif row - current[1] <= turn_rad:
                        formula = sqrt(turn_rad ** 2 - (x - mid_pt[0]) ** 2) + mid_pt[1]
                    current = [x, formula]
                    path_list, cleaned_pts, uncleaned_pts = update_list(path_list, cleaned_pts, uncleaned_pts, current)
                current[0] = ch_pt[0]
            else:
                return path_list


def draw_grid(start, vert_pts, turn_rad, direction):
    # receive and parse data on vertices
    x_val_vert = []
    y_val_vert = []
    uin = vert_pts.split()
    print(type(start))
    start_in = literal_eval(start)  # problem
    # turn_rad = int(input("enter turn_rad: "))
    # direction = input("enter direction: ")
    turn_rad = int(turn_rad)
    start = [int(start_in[0]), int(start_in[1])]
    coords = [literal_eval(coord) for coord in uin] # access by coords[0][1]
    for i in range(4):
        x_val_vert.append(coords[i][0])
        y_val_vert.append(coords[i][1])
    x_val_vert.append(coords[0][0])  # so that makes closed shape
    y_val_vert.append(coords[0][1])
    ver_points = pd.DataFrame({"x": x_val_vert, "y": y_val_vert})
    fig, ax = plt.subplots(1, 1)
    ax.set_title("Coverage Path Plan")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.plot(ver_points["x"], ver_points["y"])
    # print("hi")
    path_list = cov_plan_algo(start, ver_points, turn_rad, direction)
    ax.plot(path_list["x"], path_list["y"])
    plt.show()
    img = BytesIO()
    fig = ax.figure
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

@app.route("/logout", methods=["POST", "GET"])
def logout():  # log out by modifying session values and redirect to home
    if request.method == 'POST':  # post request
        session['logged_in'] = False
        session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)