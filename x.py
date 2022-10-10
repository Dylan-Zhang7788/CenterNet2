# 单门用来测试代码的一个py文件
import argparse
from ast import parse

parser = argparse.ArgumentParser()

parser.add_argument('--A',default='kkk',type=str)
# parser.add_argument('--A',default='ppp',type=str)

args = parser.parse_args()
args.A='ttt'

print(args)
