#!/usr/bin/env python
# coding: utf-8
#FaithHuangSihui Learning Project

with open("sequence.txt","r") as f:
    datas=f.readlines()
def is_ok(str1):
    for ch in str1:
        if(ch not in "0123456789"):
            return False
    return True
ls=[]
for data in datas:
    if("misc_feature" in data):
        x=data.strip(" \n")
        y=x.split(" ")
        if(is_ok(y[-1])):
            ls.append(int(y[-1]))
print(ls)
with open("final_result.txt","w") as f:
    for data in ls:
        f.write(str(data))
        f.write("\n")