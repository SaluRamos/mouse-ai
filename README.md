### About  
  
AI capable of moving the mouse like a human.  
  
![preview](readme/video.gif)  
  
Based on data analysis i was able to indentify this pattern  
****(ticks inside btn before click)****  
  
![alt text](readme/image.png)  
  
This is equivalent to
![alt text](readme/image2.png)

### Quickstart

- download the model .keras

for MLP:
```

import tensorflow as tf
import numpy as np
import math
from collections import deque

MODEL_PATH = "models/mouse_mlp.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
GLOBAL_COORDINATES = False
CLICK_TRESHOLD = 0.01
MAX_STEPS = 50

#pre calculated
norm_bw = bw/window_width
norm_bh = bh/window_height
target_x = bx + bw/2
target_y = by + bh/2
target_x_root = root_x + target_x
target_y_root = root_y + target_y
is_mouse_inside_btn = None
offset_x = None
offset_y = None
ticks_inside_btn = 0
next_ticks_to_click = random_ticks_inside_btn()

for _ in range(MAX_STEPS):

    if GLOBAL_COORDINATES:
        m_pos_x, m_pos_y = get_global_mouse_pos()
        btn_x_root = root_x + bx
        btn_y_root = root_y + by
        is_mouse_inside_btn = (btn_x_root + bw <= m_pos_x <= btn_x_root + bw and 
                            btn_y_root + bh <= m_pos_y <= btn_y_root + bh)
        offset_x = (target_x_root - m_pos_x)/window_width
        offset_y = (target_y_root - m_pos_y)/window_height

    else:
        m_pos_x, m_pos_y = get_mouse_pos()
        is_mouse_inside_btn = (btn_x + bw <= m_pos_x <= btn_x + bw and 
                            btn_y + bh <= m_pos_y <= btn_y + bh)
        offset_x = (target_x - m_pos_x)/window_width
        offset_y = (target_y - m_pos_y)/window_height

    if is_mouse_inside_btn:
        ticks_inside_btn += 1

    if ticks_inside_btn > 0 and not is_mouse_inside_btn:
        ticks_inside_btn = 0

    inp = tf.convert_to_tensor([[offset_x, offset_y, is_mouse_inside_btn, norm_bw, norm_bh]], dtype=tf.float32)
    mov_x_n, mov_y_n = model.predict(inp, verbose=0)

    mov_x = math.ceil(mov_x_n[0][0] * window_width)
    mov_y = math.ceil(mov_y_n[0][0] * window_height)

    mov_mouse(mov_x, mov_y) #your function that moves the mouse
    if ticks_inside_btn >= next_ticks_to_click and is_mouse_inside_btn:
        click_mouse()
        next_ticks_to_click = random_ticks_inside_btn()
        ticks_inside_btn = 0
        break

```

for RNN:
```

import tensorflow as tf
import numpy as np
import math
from collections import deque

MODEL_PATH = "models/mouse_rnn.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
GLOBAL_COORDINATES = False
CLICK_TRESHOLD = 0.2
MAX_STEPS = 50

time_steps = 20
buffer = deque(maxlen=time_steps)
for _ in range(time_steps):
    buffer.append([0.0, 0.0, 0.0, 0.0, 0.0])

#pre calculated
norm_bw = bw/window_width
norm_bh = bh/window_height
target_x = bx + bw/2
target_y = by + bh/2
target_x_root = root_x + target_x
target_y_root = root_y + target_y
is_mouse_inside_btn = None
offset_x = None
offset_y = None
ticks_inside_btn = 0
next_ticks_to_click = random_ticks_inside_btn()

for _ in range(MAX_STEPS):

    if GLOBAL_COORDINATES:
        m_pos_x, m_pos_y = get_global_mouse_pos()
        btn_x_root = root_x + bx
        btn_y_root = root_y + by
        is_mouse_inside_btn = (btn_x_root + bw <= m_pos_x <= btn_x_root + bw and 
                            btn_y_root + bh <= m_pos_y <= btn_y_root + bh)
        offset_x = (target_x_root - m_pos_x)/window_width
        offset_y = (target_y_root - m_pos_y)/window_height

    else:
        m_pos_x, m_pos_y = get_mouse_pos()
        is_mouse_inside_btn = (btn_x + bw <= m_pos_x <= btn_x + bw and 
                            btn_y + bh <= m_pos_y <= btn_y + bh)
        offset_x = (target_x - m_pos_x)/window_width
        offset_y = (target_y - m_pos_y)/window_height

    if is_mouse_inside_btn:
        ticks_inside_btn += 1

    if ticks_inside_btn > 0 and not is_mouse_inside_btn:
        ticks_inside_btn = 0

    buffer.append([offset_x, offset_y, is_mouse_inside_btn, norm_bw, norm_bh])
    inp = np.array([list(buffer)], dtype=np.float32)
    mov_x_n, mov_y_n = model.predict(inp, verbose=0)

    mov_x = math.ceil(mov_x_n[0][0] * window_width)
    mov_y = math.ceil(mov_y_n[0][0] * window_height)

    mov_mouse(mov_x, mov_y) #your function that moves the mouse
    if ticks_inside_btn >= next_ticks_to_click and is_mouse_inside_btn:
        click_mouse()
        next_ticks_to_click = random_ticks_inside_btn()
        ticks_inside_btn = 0
        break

```

### Support the project ☕

Please, dont hesitate to colaborate with the project. Give your ⭐
  
| Rede | Ícone | Endereço |
| :--- | :---: | :--- |
| **Bitcoin** | <img src="https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/btc.png" width="20"> | `bc1qg6ava2w08d5k2588edylj08ux4yuhn74yygsnr` |
| **Ethereum** | <img src="https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/eth.png" width="20"> | `0xcD36cf511646bD39Cb23f425786a4f3699CcFD2a` |
| **Solana** | <img src="https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/sol.png" width="20"> | `FKotLMzTKNbdZcKXkXsPuP1hcXGiXfScjB7qvSCQAev2` |
| **BNB Chain** | <img src="https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/bnb.png" width="20"> | `0xcD36cf511646bD39Cb23f425786a4f3699CcFD2a` |
| **TRON** | <img src="https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/trx.png" width="20"> | `TWhZLJ61uY1bo8zicwhnfS5NKuuD6BJ8D8` |

### Model
  
****Input**** = offset_x_from_target, offset_y_from_target, is_inside_btn, button_width, button_heigth  
****Output**** = mov_x, mov_y  

### Articles

[DMTG: A Human-Like Mouse Trajectory Generation Bot Based on Entropy-Controlled Diffusion Networks](https://arxiv.org/html/2410.18233v1)