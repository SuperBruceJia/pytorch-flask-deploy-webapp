#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from NERmodel import app as application

sys.path.insert(0, "/var/www/NERmodel")
sys.path.insert(0, '/usr/local/lib/python3.8/site-packages')
sys.path.insert(0, "/usr/local/lib/python3.8/bin/")

os.environ['PYTHONPATH'] = '/usr/local/lib/python3.8/bin/python'
