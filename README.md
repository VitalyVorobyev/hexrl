# Hex RL

A pet project of a reinforcement agent for the [Hex game](https://en.wikipedia.org/wiki/Hex_(board_game)). A browser GUI is implemented with [React](https://github.com/facebook/react), game driver is in python. Communication between the driver and the GUI is done via JsonRPC.

The project is at initial stage. Only random games with ASCII board implemented so far

```bash
  _____
 /     \
/   o   \_____
\       /     \
 \_____/   x   \_____
 /     \       /     \
/   .   \_____/   o   \_____
\       /     \       /     \
 \_____/   x   \_____/   .   \_____
 /     \       /     \       /     \
/   o   \_____/   o   \_____/   x   \
\       /     \       /     \       /
 \_____/   x   \_____/   x   \_____/
 /     \       /     \       /     \
/   o   \_____/   o   \_____/   o   \
\       /     \       /     \       /
 \_____/   o   \_____/   o   \_____/
 /     \       /     \       /     \
/   x   \_____/   o   \_____/   x   \
\       /     \       /     \       /
 \_____/   x   \_____/   x   \_____/
       \       /     \       /     \
        \_____/   x   \_____/   .   \
              \       /     \       /
               \_____/   x   \_____/
                     \       /     \
                      \_____/   .   \
                            \       /
                             \_____/
Blue wins (x)
```