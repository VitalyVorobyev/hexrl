# Hex RL

A pet project of a reinforcement agent for the [Hex game](https://en.wikipedia.org/wiki/Hex_(board_game)). A browser GUI is implemented with [React](https://github.com/facebook/react), game driver is in python. Communication between the driver and the GUI is done via JsonRPC.

The project is at initial stage. Only random games with ASCII board implemented so far

```bash
x x o x x . x o o x x x o   
 o x . o x x o x o x x x .  
  o . x x x o o o o x x o o 
   o . x o . x x x o x . x o
    x o o o x o . o o x . o o
     o . x x x o o o o x x x .
      o . x o x o x . o x x o x
       . x x o o o o x o o . x x
        . o o o o o . o . o o o o
         o . . x x x o o x . x x x
          x x o o . o o o x x x . o
           . . x x o o x o . x x o .
            x x o . x x o x x x o o .
Red wins (o)

81.16 ms per random 13x13 game
44.69 ms per random 11x11 game
 9.23 ms per random 7x7 game
```
