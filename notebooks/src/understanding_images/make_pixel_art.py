from pathlib import Path

import numpy as np


storage_dir = Path("notebooks", "src", "understanding_images", "data")


def _save_art(art_str, splitter_str, colors_map, art_name):
    lines = [line.strip() for line in art_str.split("\n") if line.strip()]
    array = [
        [colors_map[char] for char in
         (line.split(splitter_str) if splitter_str else line)]
        for line in lines
    ]
    array = np.array(array)
    np.save(str(Path(storage_dir, art_name)), array)



#####
# Mario

name = "art1"

# Define colors
colors = dict(
     y=(1.0, 1.0, 0.0),
     r=(1.0, 0.0, 0.0),
     s=(1.0, 0.7607843137254902, 0.5843137254901961),
     b=(0.0, 0.0, 1.0),
     h=(0.5450980392156862, 0.27058823529411763, 0.07450980392156863),
     k=(0.0, 0.0, 0.0),
     w=(1.0, 1.0, 1.0)
)

# Make art
splitter = ""
art = """
wwwrrrrrwwww
wwrrrrrrrrrw
wwhhhsskswww
whshsssksssw
whshhsssksss
wwhsssskkkkw
wwwsssssswww
wwrrbrrbrrww
wrrrbrrbrrrw
rrrrbbbbrrrr
ssrbybbybrss
sssbbbbbbsss
ssbbbbbbbbss
wwbbbwwbbbww
whhhwwwwhhhw
hhhhwwwwhhhh
"""

# Store art
_save_art(
    art_str=art, splitter_str=splitter, colors_map=colors, art_name=name
)


#####
# Mario Mushroom

name = "art2"

# Define colors
colors = dict(
     r=(1.0, 0.0, 0.0),
     b=(0.0, 0.0, 0.0),
     w=(1.0, 1.0, 1.0)
)

# Make art
splitter = ""
art = """
wwwwwbbbbbbwwwww
wwwbbrrrrwwbbwww
wwbwwrrrrwwwwbww
wbwwrrrrrrwwwwbw
wbwrrwwwwrrwwwbw
brrrwwwwwwrrrrrb
brrrwwwwwwrrwwrb
bwrrwwwwwwrwwwwb
bwwrrwwwwrrwwwwb
bwwrrrrrrrrrwwrb
bwrrbbbbbbbbrrrb
wbbbwwbwwbwwbbbw
wwbwwwbwwbwwwbww
wwwbwwwwwwwwbwww
wwwwbbbbbbbbwwww
"""

# Store art
_save_art(
    art_str=art, splitter_str=splitter, colors_map=colors, art_name=name
)


#####
# Donald duck

name = "art3"

# Define colors
colors = dict(
    w=(1., 1., 1.),
    g=(0.86, 0.86, 0.86),
    b=(0.24, 0.15, 0.67),
    k=(0., 0., 0.),
    t=(0.55, 0.71, 0.93),
    y=(0.85, 1., 0.0),
    o=(1., 0.6, 0.0),
    p=(0.95, 0.71, 0.82)
)

# Make art
splitter = ""
art = """
gggggggggbbbbbgggggg
ggggggggbbbbbbbggggg
ggggggggbbbbbbbggggg
gggggggbbbbbbbbggggg
ggggggbbbkkkkkgggggg
ggggbbbbkkkkkkgggggg
gggbbbbkwwwwwwwggggg
gggbbbkwwwwwwwwwgggg
ggkkbbwwwwtwwwwtgggg
gkkkgwwwwtttwwtttggg
kkkggwwwwtttwwtttggg
kkkggwwwwkktwwkktggg
kgkggwwwwkktwwkktggg
ggggggyywkktyykkyggg
ggggggyyyyyyyyyyyyyy
gggggggyoyyyyyyyyyyy
gggggggyoooppooygggg
ggggggggyooopooygggg
ggggggggyyoooooygggg
gggggggggyyoooyygggg
ggggggggggyyyyyggggg
"""

# Store art
_save_art(
    art_str=art, splitter_str=splitter, colors_map=colors, art_name=name
)


