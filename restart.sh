#!/bin/bash

ps aux|grep train_flower.py|grep -v grep|awk '{print $2}'|xargs kill -9

sleep 1.5

ulimit -n 65535

nohup python3 train_flower.py &

ps aux|grep train_flower.py|head -3