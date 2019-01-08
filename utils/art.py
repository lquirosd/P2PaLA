from random import shuffle, randrange


def make_maze(w=16, h=8):
    """
    Generate and show a maze, using the simple Depth-first search algorithm
    Borrowed from: https://rosettacode.org/wiki/Maze_generation#Python
    """
    vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
    ver = [["|  "] * w + ["|"] for _ in range(h)] + [[]]
    hor = [["+--"] * w + ["+"] for _ in range(h + 1)]

    def walk(x, y):
        vis[y][x] = 1

        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]:
                continue
            if xx == x:
                hor[max(y, yy)][x] = "+  "
            if yy == y:
                ver[y][max(x, xx)] = "   "
            walk(xx, yy)

    walk(randrange(w), randrange(h))

    s = ""
    ver[0][0] = "   "
    hor[-1][-2] = "+  "
    for (a, b) in zip(hor, ver):
        # s += ''.join(a + ['\n'] + b + ['\n'])
        s += "".join(a + ["\n"] + b + ["\n"])
    return s
