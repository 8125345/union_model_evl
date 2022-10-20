#!/bin/bash

kill $(pgrep -f 'transcribe_worker')
kill $(pgrep -f 'transcribe_server')
