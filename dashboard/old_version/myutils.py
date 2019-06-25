#myutils.py
# config and logging utils

import logging
import daiquiri
import configparser
from os.path import isdir, isfile
from os import mkdir, makedirs, access, R_OK

# --------------------------------------------------
# getLogPath - Get log file path from module name
# --------------------------------------------------
def getLogPath(name):
    logPath = "log\\" + name + ".log"
    return logPath


# --------------------------------------------------
# getCfgPath - Get config file path from module name
# --------------------------------------------------
def getCfgPath(name):
    cfgPath = "config\\" + name + ".cfg"
    return cfgPath


# --------------------------------------------------
# fileExists - Check if file exists
# --------------------------------------------------
def fileExists(path):
    if isfile(path) and access(path, R_OK):
        return True
    else:
        return False
            

# --------------------------------------------------
# dirExists - Check if directory exists
# --------------------------------------------------
def dirExists(path):
    if isdir(path) and access(path, R_OK):
        return True
    else:
        return False
            

# --------------------------------------------------
# createDir - Create folder
# --------------------------------------------------
def createDir(path, mode=0o777, exist_ok=True):

    try:
        makedirs(path, mode, exist_ok)

    except OSError as err:
        print("OS error: {0}".format(err))
        return False

    # print("Folder " + path + " created") 
    return True


# get configuration
def getconfig(name, section="DEFAULT"):
    config = configparser.ConfigParser()

    # Moved the check to the calling script
    # if not isdir("config"):
    #     print("Folder 'config' not found, creating...")
    #     mkdir("config")

    config.read("config\\" + name + ".cfg")
    cfg = config[section]
    return cfg


# setup logger using daiquiri
# assumes cfg has consollogging and filelogging set up
def setuplog(modulename, cfg):
    if not isdir("log"):
        print("Folder 'log' not found, creating...")
        mkdir("log")    
    daiquiri.setup(level=logging.DEBUG,
        outputs=(
        daiquiri.output.Stream(
        level=getattr(logging, cfg["consolelogging"]),
        formatter=daiquiri.formatter.ColorExtrasFormatter(
                     fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%I:%M")),
        daiquiri.output.File(
            modulename+".log",
            level=getattr(logging, cfg["filelogging"]),
            directory="log",
            formatter=daiquiri.formatter.ColorExtrasFormatter(
                    fmt="%(asctime)s [%(levelname)s]  %(message)s", datefmt="%m/%d/%Y %I:%M"))
    ))
    log = daiquiri.getLogger(modulename)
    return log

#color palettes
# TradeTeq official
TTQcolor = {
    'font': '#353535',
    'lightGrey':'#fafafa',
    'borderColour' : '#707070',
    'cream' : '#edece8',
    'lightCream' : '#F8F7F7',
    'background' : '#ffffff',
    'link' : '#3B5A95',
    'brightLink':'#FE4045',
    'marketplaceOrange' : '#F6601E',
    'warningRed' : '#ce130b',
    'Salmon' : '#F1B096',
    'victorian' : '#CEDFDD',
    'cream' : '#EFEFDC',
    'whiteGrey' : '#EDECE9',
    'eightyGrey' : '#D1D0CF',
    'blueGrey' : '#939EA9',
    'sixtyGrey' : '#8E9494',
    'navy' : '#143154',
    'darkPurple' : '#323651',
    'darkNavy' : '#0E2335',
    'darkCyan' : '#0D2B2C',
    'redBrown' : '#6B191E',
    'richBrown': '#85372B',
    'algae' : '#0E6930',
    'mutedBlue' : '#496592',
    'azureBlue' : '#155C8A',
    'electric' : '#67BDCE',
    'sky' : '#76B9D0',
    'turq' : '#1A9E78',
    'pea' : '#94CB91',
    'ocean' : '#55B7BB',
    'richOrange' : '#FB8C36',
    'richPeach' : '#F66B53',
    'yell' : '#F9EE16',
    'yellowOrange' : '#F6BD4F',
    'peach' : '#F2B097',
    'bloodRed' : '#CD130B',
    "PPTbg": '#19242F'
  }

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def brightness(color):
    b = (int(color[1:3], 16) * 299 +  int(color[3:5], 16) * 587 + int(color[5:7], 16) * 114) / 1000
    return b

def fontcolor(backgcolor):
    fontcolor = TTQcolor["font"]
    if brightness(backgcolor) < 123:
        fontcolor = TTQcolor["lightGrey"]
    return fontcolor

def colordemo():
    fig, axes = plt.subplots(7, 6, subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for idx, (key, color) in enumerate(TTQcolor.items()):
        ax = plt.subplot(7,6, idx+1)
        ax.add_patch(patches.Rectangle((0,0),1.,1.,linewidth=1,edgecolor=None,facecolor=color))
        ax.text(.50, .65, "{:}".format(key), horizontalalignment='center', 
                verticalalignment='center', transform=ax.transAxes, 
                color=fontcolor(color))
        ax.text(.50, .35, "[{:}, {:}, {:}]".format(int(color[1:3],16), 
                                                   int(color[3:5],16), 
                                                   int(color[5:7],16)), 
                horizontalalignment='center', 
                verticalalignment='center', transform=ax.transAxes, 
                color=fontcolor(color))
        ax.set_axis_off()
    

