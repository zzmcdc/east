#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:25:52 2018

@author: zhao
"""
import json
import yaml
class Param(object):
    def __init__(self):
        self.parm = {}
        
    def load_parm(self,yaml_file):
        file = open(yaml_file,'r')
        self.parm = yaml.load(file)
        file.close()

    def get_parm(self):
        return self.parm
    def __str__(self):
        return self.parm

if __name__=='__main__':
    from importlib import  import_module
    parm = Param()
    parm.load_parm('config.yaml')
    test = parm.get_parm()
    print(test)
    net = import_module(test['network.EastSymbol'])
