#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from NERmodel import app as application

sys.path.insert(0, "/var/www/NERmodel")
sys.path.insert(0, '/opt/conda/lib/python3.6/site-packages')
sys.path.insert(0, "/opt/conda/bin/")

os.environ['PYTHONPATH'] = '/opt/conda/bin/python'
